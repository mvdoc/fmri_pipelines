#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A nipype pipeline to preprocess data saved in BIDS format, based on
fmri_ants_openfmri. It performs the following steps:
1. Registration
2. Projection of the EPIs to the template space
"""
import nibabel as nb
import numpy as np
import os
from glob import glob

from nipype import config
config.enable_provenance()
from nipype.external import six
import nipype.pipeline.engine as pe
import nipype.algorithms.modelgen as model
import nipype.algorithms.rapidart as ra
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
from nipype.algorithms.misc import TSNR
from nipype.interfaces.c3 import C3dAffineTool
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
from nipype.workflows.fmri.fsl import (create_featreg_preproc,
                                       create_modelfit_workflow,
                                       create_fixed_effects_flow)
from nipype import LooseVersion
from nipype import Workflow, Node, MapNode
from nipype.interfaces import (fsl, Function, ants, freesurfer)

from nipype.interfaces.utility import Rename, Merge, IdentityInterface
from nipype.utils.filemanip import filename_to_list
from nipype.interfaces.io import DataSink, FreeSurferSource
import nipype.interfaces.freesurfer as fs


version = 0
if fsl.Info.version() and \
                LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
    version = 507

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, list_to_filename, split_filename',
           'from scipy.special import legendre'
           ]


def median(in_files):
    """Computes an average of the median of each realigned timeseries

    Parameters
    ----------

    in_files: one or more realigned Nifti 4D time series

    Returns
    -------

    out_file: a 3D Nifti file
    """
    average = None
    for idx, filename in enumerate(filename_to_list(in_files)):
        img = nb.load(filename)
        data = np.median(img.get_data(), axis=3)
        if average is None:
            average = data
        else:
            average = average + data
    median_img = nb.Nifti1Image(average/float(idx + 1),
                                img.get_affine(), img.get_header())
    filename = os.path.join(os.getcwd(), 'median.nii.gz')
    median_img.to_filename(filename)
    return filename


def create_reg_workflow(name='registration'):
    """Create a registration workflow using ANTS

    Parameters
    ----------
        name : name of workflow (default: 'registration')

    Inputs:
        inputspec.source_files :
            files (filename or list of filenames to register)
        inputspec.mean_image :
            reference image to use
        inputspec.anatomical_image :
            anatomical image to coregister to
        inputspec.target_image :
            registration target
        inputspec.target_image_brain :
            registration target after skullstripping
        inputspec.config_file :
            config file for FSL registration

    Outputs:
        outputspec.func2anat_transform :
            FLIRT transform
        outputspec.anat2target_transform :
            FNIRT transform
        outputspec.func2target_transform :
            FLIRT+FNIRT transform
        outputspec.transformed_files :
            transformed files in target space
        outputspec.transformed_mean :
            mean image in target space
        outputspec.anat2target :
            warped anatomical_image to target_image
        outputspec.mean2anat_mask :
            mask of the median EPI
        outputspec.mean2anat_mask_mni :
            mask of the median EPI in mni
        outputspec.brain :
            brain
        outputspec.anat_segmented :
            segmentation of the anatomical
        outputspec.anat_segmented_mni :
            segmentation of the anatomical in MNI
    """

    register = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=['source_files',
                    'mean_image',
                    'anatomical_image',
                    'target_image',
                    'target_image_brain',
                    'config_file']),
        name='inputspec')
    outputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=['func2anat_transform',
                    'anat2target_transform',
                    'func2target_transforms',
                    'transformed_files',
                    'transformed_mean',
                    'anat2target',
                    'mean2anat_mask',
                    'mean2anat_mask_mni',
                    'brain',
                    'anat_segmented',
                    'anat_segmented_mni']),
        name='outputspec')

    """
    Estimate the tissue classes from the anatomical image.
    """
    stripper = pe.Node(fsl.BET(frac=0.3), name='stripper')
    register.connect(inputnode, 'anatomical_image', stripper, 'in_file')
    fast = pe.Node(fsl.FAST(), name='fast')
    register.connect(stripper, 'out_file', fast, 'in_files')

    """
    Binarize the segmentation
    """
    binarize = pe.Node(fsl.ImageMaths(op_string='-nan -thr 0.5 -bin'),
                       name='binarize')
    pickindex = lambda x, i: x[i]
    register.connect(fast, ('partial_volume_files', pickindex, 2),
                     binarize, 'in_file')

    """
    Calculate rigid transform from mean image to anatomical image
    """
    mean2anat = pe.Node(fsl.FLIRT(), name='mean2anat')
    mean2anat.inputs.dof = 6
    register.connect(inputnode, 'mean_image', mean2anat, 'in_file')
    register.connect(stripper, 'out_file', mean2anat, 'reference')

    """
    Now use bbr cost function to improve the transform
    """
    mean2anatbbr = pe.Node(fsl.FLIRT(), name='mean2anatbbr')
    mean2anatbbr.inputs.dof = 6
    mean2anatbbr.inputs.cost = 'bbr'
    mean2anatbbr.inputs.schedule = os.path.join(os.getenv('FSLDIR'),
                                                'etc/flirtsch/bbr.sch')
    register.connect(inputnode, 'mean_image', mean2anatbbr, 'in_file')
    register.connect(binarize, 'out_file', mean2anatbbr, 'wm_seg')
    register.connect(inputnode, 'anatomical_image', mean2anatbbr, 'reference')
    register.connect(mean2anat, 'out_matrix_file',
                     mean2anatbbr, 'in_matrix_file')

    """
    Create a mask of the median image coregistered to the anatomical image
    """
    mean2anat_mask = Node(fsl.BET(mask=True), name='mean2anat_mask')
    register.connect(mean2anatbbr, 'out_file', mean2anat_mask, 'in_file')

    """
    Convert the BBRegister transformation to ANTS ITK format
    """
    convert2itk = pe.Node(C3dAffineTool(),
                          name='convert2itk')
    convert2itk.inputs.fsl2ras = True
    convert2itk.inputs.itk_transform = True
    register.connect(mean2anatbbr, 'out_matrix_file',
                     convert2itk, 'transform_file')
    register.connect(inputnode, 'mean_image', convert2itk, 'source_file')
    register.connect(stripper, 'out_file', convert2itk, 'reference_file')

    """
    Compute registration between the subject's structural and MNI template
    This is currently set to perform a very quick registration. However, the
    registration can be made significantly more accurate for cortical
    structures by increasing the number of iterations
    All parameters are set using the example from:
    #https://github.com/stnava/ANTs/blob/master/Scripts/newAntsExample.sh
    """
    reg = pe.Node(ants.Registration(), name='antsRegister')
    reg.inputs.output_transform_prefix = "output_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[10000, 11110, 11110]] * 2 + [[100, 30, 20]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
    reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
    reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
    reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
    reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
    reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
    reg.inputs.convergence_window_size = [20] * 2 + [5]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [False] * 2 + [True]
    reg.inputs.winsorize_lower_quantile = 0.005
    reg.inputs.winsorize_upper_quantile = 0.995
    reg.inputs.args = '--float'
    reg.inputs.output_warped_image = 'output_warped_image.nii.gz'
    reg.inputs.num_threads = 4
    reg.plugin_args = {'qsub_args': '-pe orte 4',
                       'sbatch_args': '--mem=6G -c 4'}
    register.connect(stripper, 'out_file', reg, 'moving_image')
    register.connect(inputnode, 'target_image_brain', reg, 'fixed_image')

    """
    Concatenate the affine and ants transforms into a list
    """
    merge = pe.Node(niu.Merge(2), iterfield=['in2'], name='mergexfm')
    register.connect(convert2itk, 'itk_transform', merge, 'in2')
    register.connect(reg, 'composite_transform', merge, 'in1')

    """
    Transform the mean image. First to anatomical and then to target
    """
    warpmean = pe.Node(ants.ApplyTransforms(),
                       name='warpmean')
    warpmean.inputs.input_image_type = 0
    warpmean.inputs.interpolation = 'Linear'
    warpmean.inputs.invert_transform_flags = [False, False]
    warpmean.inputs.terminal_output = 'file'

    register.connect(inputnode, 'target_image_brain',
                     warpmean, 'reference_image')
    register.connect(inputnode, 'mean_image', warpmean, 'input_image')
    register.connect(merge, 'out', warpmean, 'transforms')

    """
    Transform the remaining images. First to anatomical and then to target
    """
    warpall = pe.MapNode(ants.ApplyTransforms(),
                         iterfield=['input_image'],
                         name='warpall')
    warpall.inputs.input_image_type = 0
    warpall.inputs.interpolation = 'Linear'
    warpall.inputs.invert_transform_flags = [False, False]
    warpall.inputs.terminal_output = 'file'

    register.connect(inputnode, 'target_image_brain',
                     warpall, 'reference_image')
    register.connect(inputnode, 'source_files', warpall, 'input_image')
    register.connect(merge, 'out', warpall, 'transforms')

    """
    Transform the mask from subject space to MNI
    """
    warpmask = pe.Node(ants.ApplyTransforms(),
                       name='warpmask')
    warpmask.inputs.input_image_type = 0
    warpmask.inputs.interpolation = 'Linear'
    warpmask.inputs.invert_transform_flags = [False]
    warpmask.inputs.terminal_output = 'file'

    register.connect(inputnode, 'target_image_brain',
                     warpmask, 'reference_image')
    register.connect(mean2anat_mask, 'mask_file', warpmask, 'input_image')
    # apply only the ANTS transformation since the mask is already in
    # subject's space after BBR
    register.connect(reg, 'composite_transform', warpmask, 'transforms')

    """
    Transform the segmentations to MNI
    """
    warpsegment = pe.MapNode(ants.ApplyTransforms(),
                             iterfield=['input_image'],
                             name='warpsegment')
    warpsegment.inputs.input_image_type = 0
    warpsegment.inputs.interpolation = 'Linear'
    warpsegment.inputs.invert_transform_flags = [False, False]
    warpsegment.inputs.terminal_output = 'file'

    register.connect(inputnode, 'target_image_brain',
                     warpsegment, 'reference_image')
    register.connect(fast, 'partial_volume_files', warpsegment, 'input_image')
    register.connect(merge, 'out', warpsegment, 'transforms')

    """
    Assign all the output files
    """
    register.connect(reg, 'warped_image',
                     outputnode, 'anat2target')
    register.connect(reg, 'composite_transform',
                     outputnode, 'anat2target_transform')
    register.connect(warpmean, 'output_image',
                     outputnode, 'transformed_mean')
    register.connect(warpall, 'output_image',
                     outputnode, 'transformed_files')
    register.connect(mean2anatbbr, 'out_matrix_file',
                     outputnode, 'func2anat_transform')
    register.connect(merge, 'out',
                     outputnode, 'func2target_transforms')
    register.connect(mean2anat_mask, 'mask_file',
                     outputnode, 'mean2anat_mask')
    register.connect(stripper, 'out_file',
                     outputnode, 'brain')
    register.connect(warpmask, 'output_image',
                     outputnode, 'mean2anat_mask_mni')
    register.connect(fast, 'partial_volume_files',
                     outputnode, 'anat_segmented')
    register.connect(warpsegment, 'output_image',
                     outputnode, 'anat_segmented_mni')
    return register


def create_apply_transforms_workflow(name='bold2mni'):
    """Create a workflow to register the EPIs both to the reference EPI and MNI

    Parameters
    ----------
        name : name of workflow (default: 'bold2mni')

    Inputs:
        inputspec.source_files :
            files (filename or list of filenames to register)
        inputspec.transforms :
            transformation matrices (FLIRT+FNIRT)
        inputspec.mean_image :
            reference image to use
        inputspec.target_image :
            registration target

    Outputs:
        outputspec.transformed_files_mni :
            transformed files to MNI
        outputspec.transformed_files_anat :
            transformed files to target
    """

    register = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=['source_files',
                    'transforms',
                    'mean_image',
                    'target_image']),
        name='inputspec')
    outputnode = pe.Node(
        interface=niu.IdentityInterface(fields=[
            'transformed_files_mni',
            'transformed_files_anat']),
        name='outputspec')

    """
    Bold to MNI
    """
    warpbold = pe.MapNode(ants.ApplyTransforms(),
                          iterfield=['input_image'],
                          name='warpbold')
    warpbold.inputs.input_image_type = 3
    warpbold.inputs.interpolation = 'Linear'
    warpbold.inputs.invert_transform_flags = [False, False]
    warpbold.inputs.terminal_output = 'file'

    register.connect(inputnode, 'target_image', warpbold, 'reference_image')
    register.connect(inputnode, 'source_files', warpbold, 'input_image')
    register.connect(inputnode, 'transforms',   warpbold, 'transforms')

    """
    BOLD to individual subject anatomy
    """
    warpbold_subj = pe.MapNode(ants.ApplyTransforms(),
                               iterfield=['input_image'],
                               name='warpbold_subj')
    warpbold_subj.inputs.input_image_type = 3
    warpbold_subj.inputs.interpolation = 'Linear'
    warpbold_subj.inputs.invert_transform_flags = [False]
    warpbold_subj.inputs.terminal_output = 'file'

    # second transform is local affine one
    picksecond = lambda x: x[1]

    register.connect(inputnode, 'mean_image', warpbold_subj, 'reference_image')
    register.connect(inputnode, 'source_files', warpbold_subj, 'input_image')
    register.connect(inputnode, ('transforms', picksecond),
                     warpbold_subj, 'transforms')

    """
    Assign all the output files
    """
    register.connect(warpbold, 'output_image',
                     outputnode, 'transformed_files_mni')
    register.connect(warpbold_subj, 'output_image',
                     outputnode, 'transformed_files_anat')
    return register


def create_filter_pipeline(name='estimate_noise'):
    """
    Builds a pipeline that returns additional regressors from noise estimates.
    From nipype example rsfmri_vol_surface_preprocessing.py

    Parameters
    ----------
        name : str
            name of the pipeline (default: 'estimate_noise')

    Inputs:
        inputspec.source_files :
            files (filename or list of filenames to estimate noise from)
        inputspec.motion_parameters :
            motion parameters
        inputspec.composite_norm :
            composite norm
        inputspec.outliers :
            outliers for each volume
        inputspec.detrend_poly :
            polinomial degree to detrend
        inputspec.num_components :
            number of noise components to return
        inputspec.mask_file :
            voxel mask to return the components from (e.g., white matter)

    Outputs:
        outputspec.noise_components :
            noise components computed within the mask
    """

    # Setup some functions
    def build_filter1(motion_params, comp_norm, outliers, detrend_poly=None):
        """Builds a regressor set comprisong motion parameters, composite norm and
        outliers

        The outliers are added as a single time point column for each outlier


        Parameters
        ----------

        motion_params: a text file containing motion parameters and its derivatives
        comp_norm: a text file containing the composite norm
        outliers: a text file containing 0-based outlier indices
        detrend_poly: number of polynomials to add to detrend

        Returns
        -------
        components_file: a text file containing all the regressors
        """
        import numpy as np
        import nibabel as nb
        from scipy.special import legendre
        out_files = []
        for idx, filename in enumerate(filename_to_list(motion_params)):
            params = np.genfromtxt(filename)
            norm_val = np.genfromtxt(filename_to_list(comp_norm)[idx])
            out_params = np.hstack((params, norm_val[:, None]))
            try:
                outlier_val = np.genfromtxt(filename_to_list(outliers)[idx])
            except IOError:
                outlier_val = np.empty((0))
            for index in np.atleast_1d(outlier_val):
                outlier_vector = np.zeros((out_params.shape[0], 1))
                outlier_vector[index] = 1
                out_params = np.hstack((out_params, outlier_vector))
            if detrend_poly:
                timepoints = out_params.shape[0]
                X = np.empty((timepoints, 0))
                for i in range(detrend_poly):
                    X = np.hstack((X, legendre(
                        i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
                out_params = np.hstack((out_params, X))
            filename = os.path.join(os.getcwd(), "filter_regressor%02d.txt" % idx)
            np.savetxt(filename, out_params, fmt="%.10f")
            out_files.append(filename)
        return out_files

    def extract_noise_components(realigned_file, mask_file, num_components=5,
                                 extra_regressors=None):
        """Derive components most reflective of physiological noise

        Parameters
        ----------
        realigned_file: a 4D Nifti file containing realigned volumes
        mask_file: a 3D Nifti file containing white matter + ventricular masks
        num_components: number of components to use for noise decomposition
        extra_regressors: additional regressors to add

        Returns
        -------
        components_file: a text file containing the noise components
        """
        from scipy.linalg.decomp_svd import svd
        import numpy as np
        import nibabel as nb
        import os
        imgseries = nb.load(realigned_file)
        components = None
        for filename in filename_to_list(mask_file):
            mask = nb.load(filename).get_data()
            if len(np.nonzero(mask > 0)[0]) == 0:
                continue
            voxel_timecourses = imgseries.get_data()[mask > 0]
            voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0
            # remove mean and normalize by variance
            # voxel_timecourses.shape == [nvoxels, time]
            X = voxel_timecourses.T
            stdX = np.std(X, axis=0)
            stdX[stdX == 0] = 1.
            stdX[np.isnan(stdX)] = 1.
            stdX[np.isinf(stdX)] = 1.
            X = (X - np.mean(X, axis=0)) / stdX
            u, _, _ = svd(X, full_matrices=False)
            if components is None:
                components = u[:, :num_components]
            else:
                components = np.hstack((components, u[:, :num_components]))
        if extra_regressors:
            regressors = np.genfromtxt(extra_regressors)
            components = np.hstack((components, regressors))
        components_file = os.path.join(os.getcwd(), 'noise_components.txt')
        np.savetxt(components_file, components, fmt="%.10f")
        return components_file

    def rename(in_files, suffix=None):
        from nipype.utils.filemanip import (filename_to_list, split_filename,
                                            list_to_filename)
        out_files = []
        for idx, filename in enumerate(filename_to_list(in_files)):
            _, name, ext = split_filename(filename)
            if suffix is None:
                out_files.append(name + ('_%03d' % idx) + ext)
            else:
                out_files.append(name + suffix + ext)
        return list_to_filename(out_files)

    """
    Start of pipeline here
    """
    estimate_noise = pe.Workflow(name=name)
    inputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=[
                'source_files'
                'motion_parameters',
                'composite_norm',
                'outliers',
                'detrend_poly',
                'num_components',
                'mask_file'
            ]),
        name='inputspec')

    outputnode = pe.Node(
        interface=niu.IdentityInterface(fields=[
            'noise_components']),
        name='outputspec')

    """
    Create motion based filter to obtain residuals
    """
    make_motionbasedfilter = Node(
        Function(
            input_names=['motion_params', 'comp_norm',
                         'outliers', 'detrend_poly'],
            output_names=['out_files'],
            function=build_filter1,
            imports=imports),
        name='make_motionbasedfilter')

    estimate_noise.connect([(inputnode, make_motionbasedfilter,
                             [('motion_parameters', 'motion_params',
                               'composite_norm', 'comp_norm',
                               'outliers', 'outliers',
                               'detrend_poly', 'detrend_poly')])
                            ])

    # Link filter
    motionbasedfilter = MapNode(
        fsl.GLM(out_f_name='F_mcart.nii.gz',
                out_pf_name='pF_mcart.nii.gz',
                demean=True),
        iterfield=['in_file', 'design', 'out_res_name'],
        name='motionbasedfilter')

    estimate_noise.connect(inputnode, 'source_files',
                           motionbasedfilter, 'in_file')
    estimate_noise.connect(
        inputnode, ('source_files', rename, '_filtermotart'),
        motionbasedfilter, 'out_res_name')
    estimate_noise.connect(make_motionbasedfilter, 'out_files',
                           motionbasedfilter, 'design')

    """
    Get noise components on residuals within the provided mask
    """
    make_compcorrfilter = MapNode(
        Function(
            input_names=[
                'realigned_file',
                'mask_file',
                'num_components',
                'extra_regressors'],
            output_names=['out_files'],
            function=extract_noise_components,
            imports=imports),
        iterfield=['realigned_file', 'extra_regressors'],
        name='make_compcorrfilter')

    estimate_noise.connect(inputnode, 'num_components',
                           make_compcorrfilter, 'num_components')
    estimate_noise.connect(make_motionbasedfilter, 'out_files',
                           make_compcorrfilter, 'extra_regressors')
    estimate_noise.connect(motionbasedfilter, 'out_res',
                           make_compcorrfilter, 'realigned_file')
    # pick only what is in the mask
    estimate_noise.connect(inputnode, 'mask_file',
                           make_compcorrfilter, 'mask_file')

    # return what we need
    estimate_noise.connect(make_compcorrfilter, 'out_files',
                           outputnode, 'noise_components')

    return estimate_noise


def get_subjectinfo(subject_id, base_dir, task_id, session_id=''):
    """
    Get info for a given subject

    Parameters
    -----------
    subject_id : str
        Subject identifier (e.g., sub-01)
    base_dir : str
        Path to base directory of the dataset
    task_id : str
        Which task to process (e.g., task-facelocalizer)
    session_id : str or None
        Which session to process (e.g., ses-fmri01)

    Returns
    -------
    run_ids : list of ints
        Run numbers
    TR : float
        Repetition time
    """
    pass


def preprocess(data_dir, subject=None, task_id=None, output_dir=None,
               subj_prefix='*', hpcutoff=120., fwhm=6.0,
               num_noise_components=5):
    """Preprocesses a BIDS dataset

    Parameters
    ----------

    data_dir : str
        Path to the base data directory
    subject : str
        Subject id to preprocess. If None, all will be preprocessed
    task_id : str
        Task to process
    output_dir : str
        Output directory
    subj_prefix : str
    hpcutoff : float
        high pass cutoff in seconds
    fwhm : float
        smoothing parameter
    num_noise_components : int
        number of PCs of timeseries in ventricles to output as additional
        noise regressors
    """

    """
    Load nipype workflows
    """
    preproc = create_featreg_preproc(whichvol='first')
    registration = create_reg_workflow()
    reslice_bold = create_apply_transforms_workflow()
    estimate_noise = create_filter_pipeline()

    """
    Remove the plotting connection so that plot iterables don't propagate
    to the model stage
    """
    preproc.disconnect(preproc.get_node('plot_motion'), 'out_file',
                       preproc.get_node('outputspec'), 'motion_plots')

    """
    Set up bids data specific components
    """
    # XXX: check this
    subjects = sorted([path.split(os.path.sep)[-1] for path in
                       glob(os.path.join(data_dir, subj_prefix))])

    infosource = pe.Node(
        niu.IdentityInterface(
            fields=['subject_id', 'task_id']),
        name='infosource')

    if len(subject) == 0:
        infosource.iterables = [('subject_id', subjects),
                                ('task_id', task_id)]
    else:
        infosource.iterables = [('subject_id',
                                 [subjects[subjects.index(subj)]
                                  for subj in subject]),
                                ('task_id', task_id)]

    subjinfo = pe.Node(
        niu.Function(
            input_names=['subject_id', 'base_dir',
                         'task_id'],
            # XXX: change this according to get_subjectinfo
            output_names=['run_id', 'conds', 'TR'],
            function=get_subjectinfo),
        name='subjectinfo')
    subjinfo.inputs.base_dir = data_dir

    """
    Set up DataGrabber to return anat, bold, and behav
    """
    # XXX: rename behav to events? or do I even need it?
    datasource = pe.Node(
        nio.DataGrabber(
            infields=['subject_id', 'run_id', 'task_id'],
            outfields=['anat', 'bold', 'behav']),
        name='datasource')

    datasource.inputs.base_directory = data_dir
    datasource.inputs.template = '*'

    datasource.inputs.field_template = {
        'anat': 'sub-%s/anat/sub-%s_T1w.nii.gz',
        'bold': 'sub-%s/func/sub-%s_task-%s_run-*_bold.nii.gz',
    }

    datasource.inputs.template_args = {
        'anat': [['subject_id', 'subject_id']],
        'bold': [['subject_id', 'subject_id', 'task_id']],
    }

    datasource.inputs.sort_filelist = True

    """
    Create meta workflow
    """
    wf = pe.Workflow(name='bids_preprocess')
    wf.connect(infosource, 'subject_id', subjinfo, 'subject_id')
    wf.connect(infosource, 'task_id', subjinfo, 'task_id')
    wf.connect(infosource, 'subject_id', datasource, 'subject_id')
    wf.connect(infosource, 'task_id', datasource, 'task_id')
    wf.connect(subjinfo, 'run_id', datasource, 'run_id')
    # connect bold to preprocessing pipeline
    wf.connect(datasource, 'bold', preproc, 'inputspec.func')

    """
    Get highpass information for preprocessing
    """
    def get_highpass(TR, hpcutoff):
        return hpcutoff / (2 * TR)
    gethighpass = pe.Node(niu.Function(
        input_names=['TR', 'hpcutoff'],
        output_names=['highpass'],
        function=get_highpass),
        name='gethighpass')

    wf.connect(subjinfo, 'TR', gethighpass, 'TR')
    wf.connect(gethighpass, 'highpass', preproc, 'inputspec.highpass')

    """
    QA: Compute TSNR on realigned data regressing polynomials up to order 2
    """
    tsnr = MapNode(
        TSNR(regress_poly=2),
        iterfield=['in_file'],
        name='tsnr')
    wf.connect(preproc, 'outputspec.realigned_files', tsnr, 'in_file')

    # Compute the median image across runs
    calc_median = Node(
        Function(input_names=['in_files'],
                 output_names=['median_file'],
                 function=median,
                 imports=imports),
        name='median')
    wf.connect(tsnr, 'detrended_file', calc_median, 'in_files')

    """
    Setup artifact detection
    """
    art = pe.MapNode(
        interface=ra.ArtifactDetect(
            use_differences=[True, False],
            use_norm=True,
            norm_threshold=1,
            zintensity_threshold=3,
            parameter_source='FSL',
            mask_type='file'),
        iterfield=['realigned_files', 'realignment_parameters', 'mask_file'],
        name="art")
    wf.connect([(preproc, art,
                 [('outputspec.realigned_files', 'realigned_files'),
                  ('outputspec.motion_parameters', 'realignment_parameters'),
                  ('outputspec.mask', 'mask_file')])
                ])

    """
    Register anatomical to template
    """
    wf.connect(datasource, 'anat',
               registration, 'inputspec.anatomical_image')
    registration.inputs.inputspec.target_image = \
        fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')
    registration.inputs.inputspec.target_image_brain = \
        fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
    registration.inputs.inputspec.config_file = 'T1_2_MNI152_2mm'
    # Use median tSNR image as mean_image for registration
    wf.connect(calc_median, 'median_file',
               registration, 'inputspec.mean_image')

    """
    QA: Check alignment of bet to anatomical
    """
    slicer = pe.Node(
        fsl.Slicer(image_width=1300, sample_axial=12,
                   nearest_neighbour=True, threshold_edges=-0.1),
        name='slicer')
    wf.connect(datasource, 'anat', slicer, 'in_file')
    wf.connect(registration, 'outputspec.brain', slicer, 'image_edges')

    """
    QA: Check alignment of mean bold image to anatomical
    """
    slicer_bold = pe.Node(
        fsl.Slicer(image_width=1300, sample_axial=8,
                   nearest_neighbour=True, threshold_edges=0.1),
        name='slicer_bold')
    wf.connect(registration, 'inputspec.target_image', slicer_bold, 'in_file')
    wf.connect(registration, 'outputspec.transformed_mean',
               slicer_bold, 'image_edges')

    """
    Reslice BOLD into subject-specific and MNI space
    """
    wf.connect(preproc, 'outputspec.highpassed_files',
               reslice_bold, 'inputspec.source_files')
    wf.connect([
        (registration, reslice_bold,
         # use the same mean_image used in registration
         [('inputspec.mean_image', 'inputspec.mean_image'),
          ('outputspec.func2target_transforms', 'inputspec.transforms'),
          ('inputspec.target_image', 'inputspec.target_image')])
               ])

    """
    Compute noise estimates within the white matter mask
    """
    # Use this function to select which mask to use
    def selectindex(files, idx):
        from nipype.utils.filemanip import filename_to_list, list_to_filename
        out_array = np.array(filename_to_list(files))[idx].tolist()
        return list_to_filename(out_array)

    estimate_noise.inputs.detrend_poly = 2
    estimate_noise.inputs.num_components = num_noise_components
    wf.connect(preproc, 'outputspec.motion_parameters',
               estimate_noise, 'inputspec.motion_parameters')
    wf.connect(art, 'norm_files',
               estimate_noise, 'inputspec.composite_norm')
    wf.connect(art, 'outlier_files',
               estimate_noise, 'inputspec.outliers')
    wf.connect(reslice_bold, 'transformed_files_mni',
               estimate_noise, 'inputspec.source_files')
    # take only white matter mask
    wf.connect(
        registration, ('outputspec.anat_segmented_mni', selectindex, [2]),
        estimate_noise, 'inputspec.mask_file')


    """
    TODO: Connect to a datasink
    """
