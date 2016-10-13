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
    """Create a FEAT preprocessing workflow

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
    register.connect(reg, 'composite_transform',
                     outputnode, 'anat2target_transform')
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


def preprocess(data_dir, subject=None, output_dir=None, subj_prefix='*',
               hpcutoff=120., fwhm=6.0, num_noise_components=5):
    """Preprocesses a BIDS dataset

    Parameters
    ----------

    data_dir : str
        Path to the base data directory
    subject : str
        Subject id to preprocess. If None, all will be preprocessed
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

    """
    Remove the plotting connection so that plot iterables don't propagate
    to the model stage
    """
    preproc.disconnect(preproc.get_node('plot_motion'), 'out_file',
                       preproc.get_node('outputspec'), 'motion_plots')

    """
    Create meta workflow
    """
    wf = pe.Workflow(name='bids_preprocess')
    wf.connect(infosource, 'subject_id', subjinfo, 'subject_id')
    wf.connect(infosource, 'model_id', subjinfo, 'model_id')
    wf.connect(infosource, 'task_id', subjinfo, 'task_id')
    wf.connect(infosource, 'subject_id', datasource, 'subject_id')
    wf.connect(infosource, 'model_id', datasource, 'model_id')
    wf.connect(infosource, 'task_id', datasource, 'task_id')
    wf.connect(subjinfo, 'run_id', datasource, 'run_id')
    wf.connect([(datasource, preproc, [('bold', 'inputspec.func')]),
                ])

    def get_highpass(TR, hpcutoff):
        return hpcutoff / (2 * TR)
    gethighpass = pe.Node(niu.Function(input_names=['TR', 'hpcutoff'],
                                       output_names=['highpass'],
                                       function=get_highpass),
                          name='gethighpass')
    wf.connect(subjinfo, 'TR', gethighpass, 'TR')
    wf.connect(gethighpass, 'highpass', preproc, 'inputspec.highpass')
