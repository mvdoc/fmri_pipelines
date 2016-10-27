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
import nipype.pipeline.engine as pe
import nipype.algorithms.rapidart as ra
from nipype.algorithms.misc import TSNR, Gunzip
from nipype.interfaces.c3 import C3dAffineTool
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
from nipype.workflows.fmri.fsl import create_featreg_preproc
from nipype import LooseVersion
from nipype import Workflow, Node, MapNode
from nipype.interfaces import (fsl, Function, ants, afni)

from nipype.utils.filemanip import filename_to_list
from nipype.interfaces.io import DataSink, FreeSurferSource
import nipype.interfaces.freesurfer as fs

version = 0
if fsl.Info.version() and \
                LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
    version = 507

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')

imports = ['import numpy as np',
           'import os',
           'import nibabel as nb',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, '
           'list_to_filename, split_filename',
           'from scipy.special import legendre',
           'from scipy.linalg.decomp_svd import svd'
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


def create_fieldmapcorrection_workflow(name='fmapcorrection'):
    """Create a workflow to do fieldmap correction with SIEMENS scanners

    Parameters
    ----------
        name : name of workflow (default: 'fmapcorrection')

    Inputs:
        inputspec.source_files :
            files (filename or list of filenames to fieldmap correct)
        inputspec.magnitude_file :
            file containing magnitude information
        inputspec.phase_file :
            file containing phase information
        inputspec.delta_TE :
            echo time difference of the fieldmap sequence in ms
        inputspec.dwell_time :
            dwell time (aka Echo Spacing)


    Outputs:
        outputspec.unwarped_files :
            unwarped source files
    """

    fmapcorrect = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=['source_files',
                    'magnitude_file',
                    'phase_file',
                    'delta_TE',
                    'dwell_time']),
        name='inputspec')
    outputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=['unwarped_files']),
        name='outputspec')

    """
    Perform skullstripping on magnitude
    """
    bet = pe.Node(fsl.BET(), name='betfmap')
    fmapcorrect.connect(inputnode, 'magnitude_file',
                        bet, 'in_file')

    """
    Gunzip phasediff for fsl_prepare_fieldmap
    """
    gunzip = pe.Node(Gunzip(), name='gunzip')
    fmapcorrect.connect(inputnode, 'phase_file',
                        gunzip, 'in_file')

    """
    Prepare fieldmap using fsl_prepare_fieldmap
    """
    prepare = pe.Node(fsl.PrepareFieldmap(), name='preparefmap')
    fmapcorrect.connect(bet, 'out_file',
                        prepare, 'in_magnitude')
    fmapcorrect.connect(gunzip, 'out_file',
                        prepare, 'in_phase')
    fmapcorrect.connect(inputnode, 'delta_TE',
                        prepare, 'delta_TE')

    """
    Apply fieldmap correction
    """
    fugue = pe.MapNode(fsl.FUGUE(),
                       iterfield=['in_file'],
                       name='fugue')
    fmapcorrect.connect(prepare, 'out_fieldmap',
                        fugue, 'fmap_in_file')
    fmapcorrect.connect([(inputnode, fugue,
                         [('dwell_time', 'dwell_time'),
                          ('source_files', 'in_file')])
                         ])

    """
    Connect output
    """
    fmapcorrect.connect(fugue, 'unwarped_file',
                        outputnode, 'unwarped_files')

    return fmapcorrect


def create_registration_workflow(name='registration'):
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
    stripper = pe.Node(afni.SkullStrip(), name='stripper')
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
#    warpall = pe.MapNode(ants.ApplyTransforms(),
#                         iterfield=['input_image'],
#                         name='warpall')
#    warpall.inputs.input_image_type = 0
#    warpall.inputs.interpolation = 'Linear'
#    warpall.inputs.invert_transform_flags = [False, False]
#    warpall.inputs.terminal_output = 'file'
#
#    register.connect(inputnode, 'target_image_brain',
#                     warpall, 'reference_image')
#    register.connect(inputnode, 'source_files', warpall, 'input_image')
#    register.connect(merge, 'out', warpall, 'transforms')

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
    #register.connect(warpall, 'output_image',
    #                 outputnode, 'transformed_files')
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


def create_estimatenoise_workflow(name='estimate_noise'):
    """
    Builds a workflow that returns additional regressors from noise estimates.
    From nipype example rsfmri_vol_surface_preprocessing.py

    Parameters
    ----------
        name : str
            name of the workflow (default: 'estimate_noise')

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
                'source_files',
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
                             [('motion_parameters', 'motion_params'),
                              ('composite_norm', 'comp_norm'),
                              ('outliers', 'outliers'),
                              ('detrend_poly', 'detrend_poly')])
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
    estimate_noise.connect(make_motionbasedfilter, 'out_files',
                           motionbasedfilter, 'design')
    estimate_noise.connect(
        inputnode, ('source_files', rename, '_filtermotart'),
        motionbasedfilter, 'out_res_name')

    """
    Get noise components on residuals within the provided mask
    """
    make_compcorrfilter = MapNode(
        Function(
            input_names=[
                'realigned_file',
                'mask_file',
                'num_components',
                'extra_regressors'
                ],
            output_names=['out_files'],
            function=extract_noise_components,
            imports=imports),
        iterfield=['realigned_file', 'extra_regressors'],
        name='make_compcorrfilter')

    estimate_noise.connect(motionbasedfilter, 'out_res',
                           make_compcorrfilter, 'realigned_file')
    # pick only what is in the mask
    estimate_noise.connect(inputnode, 'mask_file',
                           make_compcorrfilter, 'mask_file')
    estimate_noise.connect(inputnode, 'num_components',
                           make_compcorrfilter, 'num_components')
    estimate_noise.connect(make_motionbasedfilter, 'out_files',
                           make_compcorrfilter, 'extra_regressors')

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
        Subject identifier (e.g., sid000001)
    base_dir : str
        Path to base directory of the dataset
    task_id : str
        Which task to process (e.g., facelocalizer)
    session_id : str or None
        Which session to process (e.g., ses-fmri01)

    Returns
    -------
    run_ids : list of ints
        Run numbers
    TR : float
        Repetition time
    delta_TE : float
        difference of echo times for fieldmap
    dwell_time : float
        Effective Echo Spacing
    """
    import os
    import re
    import json
    from glob import glob

    subject_funcdir = os.path.join(
        base_dir,
        '{0}'.format(subject_id),
        'func',
    )

    subject_fmapdir = subject_funcdir.replace('func', 'fmap')

    # get run ids
    runs_template = os.path.join(
        subject_funcdir,
        '{0}_task-{1}_*run-*_bold.nii*'.format(subject_id, task_id)
    )
    runs = glob(runs_template)
    run_ids = sorted([int(re.findall('run-([0-9]*)', r)[0]) for r in runs])

    # Note: in DBIC standards we'll have a json file for every run, thus
    # assume that each task will have the same RT
    run_jsons = glob(
        os.path.join(
            subject_funcdir,
            '{0}_task-{1}_*run-*_bold.json'.format(subject_id, task_id)
        )
    )
    if len(runs) != len(run_jsons):
        raise ValueError('Got {0} runs but {1} '
                         'json files'.format(len(runs), len(run_jsons)))

    with open(run_jsons[0], 'rt') as f:
        dataset_info = json.load(f)

    # load fmap phasediff json file to compute delta TE
    fmap_jsons = glob(
        os.path.join(
            subject_fmapdir,
            '{0}_*run-*_phasediff.json'.format(subject_id)
        )
    )
    # XXX: use only the first one atm
    with open(fmap_jsons[0], 'rt') as f:
        phasediff_info = json.load(f)
    delta_TE = (phasediff_info['EchoTime1'] - phasediff_info['EchoTime2'])*1000

    return run_ids, dataset_info['RepetitionTime'], delta_TE,\
        dataset_info['EffectiveEchoSpacing']


def preprocess_pipeline(data_dir, subject=None, task_id=None, output_dir=None,
                        subj_prefix='sub-*', hpcutoff=120., fwhm=6.0,
                        num_noise_components=5):
    """
    Preprocesses a BIDS dataset

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
    registration = create_registration_workflow()
    reslice_bold = create_apply_transforms_workflow()
    estimate_noise = create_estimatenoise_workflow()
    fmapcorr = create_fieldmapcorrection_workflow()

    """
    Remove the plotting connection so that plot iterables don't propagate
    to the model stage
    """
    #preproc.disconnect(preproc.get_node('plot_motion'), 'out_file',
    #                   preproc.get_node('outputspec'), 'motion_plots')

    """
    Set up bids data specific components
    """
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
            output_names=['run_id', 'TR', 'delta_TE', 'dwell_time'],
            function=get_subjectinfo),
        name='subjectinfo')
    subjinfo.inputs.base_dir = data_dir

    """
    Set up DataGrabber to return anat and bold
    """
    datasource = pe.Node(
        nio.DataGrabber(
            infields=['subject_id', 'run_id', 'task_id'],
            outfields=['anat', 'bold', 'fmap_magnitude', 'fmap_phase']),
        name='datasource')

    datasource.inputs.base_directory = data_dir
    datasource.inputs.template = '*'

    datasource.inputs.field_template = {
        'anat': '%s/anat/%s_T1w.nii*',
        'bold': '%s/func/%s_task-%s_*run-*_bold.nii*',
        'fmap_magnitude': '%s/fmap/%s_*run-*_magnitude*.nii*',
        'fmap_phase': '%s/fmap/%s_*run-*_phasediff.nii*'
    }

    datasource.inputs.template_args = {
        'anat': [['subject_id', 'subject_id']],
        'bold': [['subject_id', 'subject_id', 'task_id']],
        'fmap_magnitude': [['subject_id', 'subject_id']],
        'fmap_phase': [['subject_id', 'subject_id']]
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

    """
    Perform fieldmap correction
    """
    # get the first magnitude only
    def pickfirst(x):
        return x[0]
    wf.connect(datasource, ('fmap_magnitude', pickfirst),
               fmapcorr, 'inputspec.magnitude_file')
    wf.connect(datasource, 'fmap_phase',
               fmapcorr, 'inputspec.phase_file')
    wf.connect(subjinfo, 'delta_TE',
               fmapcorr, 'inputspec.delta_TE')
    wf.connect(subjinfo, 'dwell_time',
               fmapcorr, 'inputspec.dwell_time')
    # connect bold to fmap
    wf.connect(datasource, 'bold', fmapcorr, 'inputspec.source_files')

    """
    Run preprocessing
    """
    # connect warped bold to preprocessing pipeline
    wf.connect(fmapcorr, 'outputspec.unwarped_files',
               preproc, 'inputspec.func')

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
    QA: Check skull stripping on anatomical
    """
    slicer_skullstrip = pe.Node(
        fsl.Slicer(image_width=1300, all_axial=True, #sample_axial=12,
                   nearest_neighbour=True, threshold_edges=-0.1),
        name='slicer')
    wf.connect(datasource, 'anat', slicer_skullstrip, 'in_file')
    wf.connect(registration, 'outputspec.brain',
               slicer_skullstrip, 'image_edges')

    """
    QA: Check alignment of mean bold image to anatomical
    """
    slicer_bold = pe.Node(
        fsl.Slicer(image_width=1300, all_axial=True, #sample_axial=8,
                   nearest_neighbour=True, threshold_edges=0.1),
        name='slicer_bold')
    wf.connect(registration, 'inputspec.target_image', slicer_bold, 'in_file')
    wf.connect(registration, 'outputspec.transformed_mean',
               slicer_bold, 'image_edges')

    """
    QA: Check ANTS registration of anatomical to target template
    """
    slicer_ants = pe.Node(
        fsl.Slicer(image_width=1300, all_axial=True, #sample_axial=12,
                   nearest_neighbour=True, threshold_edges=0.1),
        name='slicer_ants')
    wf.connect(registration, 'inputspec.target_image', slicer_ants, 'in_file')
    wf.connect(registration, 'outputspec.anat2target',
               slicer_ants, 'image_edges')

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
        import numpy as np
        from nipype.utils.filemanip import filename_to_list, list_to_filename
        out_array = np.array(filename_to_list(files))[idx].tolist()
        return list_to_filename(out_array)

    estimate_noise.inputs.inputspec.detrend_poly = 2
    estimate_noise.inputs.inputspec.num_components = num_noise_components
    wf.connect(preproc, 'outputspec.motion_parameters',
               estimate_noise, 'inputspec.motion_parameters')
    wf.connect(art, 'norm_files',
               estimate_noise, 'inputspec.composite_norm')
    wf.connect(art, 'outlier_files',
               estimate_noise, 'inputspec.outliers')
    wf.connect(reslice_bold, 'outputspec.transformed_files_mni',
               estimate_noise, 'inputspec.source_files')
    # take only white matter mask
    wf.connect(
        registration, ('outputspec.anat_segmented_mni', selectindex, [2]),
        estimate_noise, 'inputspec.mask_file')

    """
    Connect to a datasink
    """
    # Setup substitutions for filenames
    def get_subs(subject_id, run_id, task_id):
        subs = list()
        subs.append(('_subject_id_{0}_'.format(subject_id),
                     '{0}'.format(subject_id)))
        subs.append(('_dtype_mcf_mask_smooth_mask_gms_tempfilt_maths_trans',
                     ''))
        subs.append(('{0}task_id_{1}/'.format(subject_id, task_id),
                     ''))
        subs.append(('dtype_mcf_bet_thresh_dil', 'mask'))
        subs.append(('_bold_dtype_mcf.nii.gz', ''))

        art_template = '{subject_id}_task-{task_id}_run-{run_id:02d}'
        for i, run_num in enumerate(run_id):
            # art
            subs_art = {'art': 'art',
                        'global_intensity': 'globalintensity',
                        'norm': 'norm'}
            for fromwhat, towhat in subs_art.iteritems():
                this_templ = art_template.format(subject_id=subject_id,
                                                 task_id=task_id,
                                                 run_id=run_num)
                suffix = '' if fromwhat == 'art' else '_' + towhat
                subs.append(('_art{0}/'.format(i) + fromwhat + '.' +
                             this_templ + '_bold_dtype_mcf',
                             this_templ + suffix))
            # warpbold
            subs.append(('_warpbold{0}/'.format(i),
                         ''))
            # warpbold_subj
            subs.append(('_warpbold_subj{0}/'.format(i),
                         ''))
            # motion
            subs.append(('_realign{0}/'.format(i),
                         ''))
            # tsnr
            subs.append(('_tsnr{0}/tsnr'.format(i),
                         '{0}_task-{1}_run-{2:02d}_tsnr'.format(subject_id,
                                                                task_id,
                                                                run_num)))
            # dilate mask
            subs.append(('_dilatemask{0}/'.format(i),
                         ''))

            # noisecomp
            subs.append((
                '_make_compcorrfilter{0}/noise_components'.format(i),
                '{0}_task-{1}_run-{2:02d}_noisecomponents'.format(subject_id,
                                                                  task_id,
                                                                  run_num)
            ))
        # slicer images
        subs.append(('{0}_T1w.png'.format(subject_id),
                     '{0}_T1w_brain.png'.format(subject_id)))
        subs.append(('mean2mni/MNI152_T1_2mm.png',
                     '{0}_task-{1}_mean2mni.png'.format(subject_id, task_id)))
        subs.append(('anat2mni/MNI152_T1_2mm.png',
                     '{0}_task-{1}_anat2mni.png'.format(subject_id, task_id)))
        # warpsegment
        for i in range(3):
            subs.append(('_warpsegment{0}'.format(i), '/'))
        subs.append(('_trans.nii', '_mni.nii'))

        # skullstrip
        subs.append(('skullstrip', 'brain'))

        # warped anatomy
        subs.append(('output_warped_image',
                     '{0}_task-{1}_T1w_brain_mni'.format(subject_id, task_id)))

        # median image mask
        subs.append(('median_flirt_brain_mask',
                     '{0}_task-{1}_brain_bold_mask'.format(subject_id,
                                                           task_id)))
        subs.append(('median',
                    '{0}_task-{1}_median'.format(subject_id, task_id)))

        return subs

    # Make substitution node
    subsgen = pe.Node(
        niu.Function(
            input_names=['subject_id',
                         'run_id',
                         'task_id'],
            output_names=['substitutions'],
            function=get_subs),
        name='subsgen')
    wf.connect(subjinfo, 'run_id', subsgen, 'run_id')

    # Now make datasink node
    datasink = pe.Node(interface=nio.DataSink(), name="datasink")
    wf.connect(infosource, 'subject_id', datasink, 'container')
    wf.connect(infosource, 'subject_id', subsgen, 'subject_id')
    wf.connect(infosource, 'task_id', subsgen, 'task_id')
    wf.connect(subsgen, 'substitutions', datasink, 'substitutions')

    # Pre-processing output
    wf.connect([(preproc, datasink,
                 [('outputspec.motion_parameters', 'qa.motion'),
                  ('outputspec.motion_plots', 'qa.motion.plots'),
                  ('outputspec.mask', 'qa.mask')]
                 )])

    # registration output
    wf.connect([(registration, datasink,
                 [('outputspec.mean2anat_mask', 'mask.mean2anat'),
                  ('outputspec.mean2anat_mask_mni', 'mask.mean2mni'),
                  ('outputspec.anat2target', 'qa.anat2mni'),
                  ('outputspec.transformed_mean', 'mean.mni'),
                  ('outputspec.func2anat_transform', 'xfm.mean2anat'),
                  ('outputspec.func2target_transforms', 'xfm.mean2mni'),
                  ('outputspec.anat2target_transform', 'xfm.anat2mni'),
                  ('outputspec.anat_segmented', 'segm'),
                  ('outputspec.anat_segmented_mni', 'segm.mni')])
                ])
    # artifact detection output
    wf.connect([(art, datasink,
                 [('norm_files', 'qa.art.@norm'),
                  ('intensity_files', 'qa.art.@intensity'),
                  ('outlier_files', 'qa.art.@outlier_files')]
                 )])
    # tsnr
    wf.connect(tsnr, 'tsnr_file', datasink, 'qa.tsnr.@map')
    # median tsnr
    wf.connect(calc_median, 'median_file', datasink, 'mean')
    # slicer
    wf.connect(slicer_skullstrip, 'out_file', datasink, 'qa.plots.@skullstrip')
    wf.connect(slicer_bold, 'out_file', datasink, 'qa.plots.mean2mni')
    wf.connect(slicer_ants, 'out_file', datasink, 'qa.plots.anat2mni')
    # resliced bolds
    wf.connect([(reslice_bold, datasink,
                 [('outputspec.transformed_files_mni', 'func.mni'),
                  ('outputspec.transformed_files_anat', 'func')])
                ])
    # noise components
    wf.connect(estimate_noise, 'outputspec.noise_components',
               datasink, 'noisecomp')

    """
    Set processing parameters and return workflow
    """
    preproc.inputs.inputspec.fwhm = fwhm
    gethighpass.inputs.hpcutoff = hpcutoff
    datasink.inputs.base_directory = output_dir

    return wf

"""
Make command line parser
"""
if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='preprocess.py',
                                     description=__doc__)
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument('-s', '--subject', default=[],
                        nargs='+', type=str,
                        help="Subject name (e.g. 'sid00001')")
    parser.add_argument('-x', '--subjectprefix', default='sub-',
                        help="Subject prefix" + defstr)
    parser.add_argument('-t', '--task', default='',
                        type=str, help="Task name" + defstr)
    parser.add_argument('--hpfilter', default=120., type=float,
                        help="High pass filter cutoff (in secs)" + defstr)
    parser.add_argument('--fwhm', default=6.,
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--write-graph", default="",
                        help="Do not run, just write the graph to "
                             "specified file")

    args = parser.parse_args()
    outdir = args.outdir
    work_dir = os.getcwd()
    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    if outdir:
        outdir = os.path.abspath(outdir)
    else:
        outdir = os.path.join(work_dir, 'output')

    wf = preprocess_pipeline(data_dir=os.path.abspath(args.datasetdir),
                             subject=args.subject,
                             task_id=[args.task],
                             subj_prefix=args.subjectprefix + '*',
                             output_dir=outdir,
                             hpcutoff=args.hpfilter,
                             fwhm=args.fwhm,
                             )
    # wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.base_dir = work_dir
    if args.write_graph:
        wf.write_graph(args.write_graph)
    elif args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
