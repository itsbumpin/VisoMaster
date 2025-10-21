from app.helpers.typing_helper import LayoutDictTypes

FACE_EDITOR_LAYOUT_DATA: LayoutDictTypes = {
    '': {
        'FaceEditorCropScaleDecimalSlider': {
            'level': 1,
            'label': 'Crop Scale',
            'min_value': '1.50',
            'max_value': '4.00',
            'default': '2.50',
            'step': 0.05,
            'decimals': 2,
            'help': 'Changes source crop scale. Increase the value to capture the face more distantly. 2.2 scale factor for cropping driving video.'
        },
        'FaceEditorVYRatioDecimalSlider': {
            'level': 1,
            'label': 'VY Ratio',
            'min_value': '-0.200',
            'max_value': '0.200',
            'default': '-0.125',
            'step': 0.001,
            'decimals': 3,
            'help': 'Changes the vy ratio for crop scale. Increase the value to capture the face more distantly. -0.1 factor for cropping driving video.'
        },
        'FaceEditorBlurAmountSlider': {
            'level': 1,
            'label': 'Blur Amount',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'help': 'Blur amount.'
        },
        'FaceEditorEnableToggle': {
            'level': 1,
            'label': 'Enable Face Pose/Expression Editor',
            'default': False,
            'help': 'Enable Face Pose/Expression Editor.'
        },
        'FaceEditorTypeSelection': {
            'level': 2,
            'label': 'Face Editor Type',
            #'options': ['Human-Face', 'Animal-Face'],
            'options': ['Human-Face'],
            'default': 'Human-Face',
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Select the target type to be edited in Face Editor.'
        },
        'EyesOpenRatioDecimalSlider': {
            'level': 2,
            'label': 'Eyes Close <--> Open Ratio',
            'min_value': '-0.80',
            'max_value': '0.80',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the opening of the eyes.'
        },
        'LipsOpenRatioDecimalSlider': {
            'level': 2,
            'label': 'Lips Close <--> Open Ratio',
            'min_value': '-0.80',
            'max_value': '0.80',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the opening of the lips.'
        },
        'HeadPitchSlider': {
            'level': 2,
            'label': 'Head Pitch',
            'min_value': '-15',
            'max_value': '15',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the opening of the lips.'
        },
        'HeadYawSlider': {
            'level': 2,
            'label': 'Head Yaw',
            'min_value': '-15',
            'max_value': '15',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the head yaw.'
        },
        'HeadRollSlider': {
            'level': 2,
            'label': 'Head Roll',
            'min_value': '-15',
            'max_value': '15',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the head roll.'
        },
        'XAxisMovementDecimalSlider': {
            'level': 2,
            'label': 'X-Axis Movement',
            'min_value': '-0.19',
            'max_value': '0.19',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the head direction x-axis.'
        },
        'YAxisMovementDecimalSlider': {
            'level': 2,
            'label': 'Y-Axis Movement',
            'min_value': '-0.19',
            'max_value': '0.19',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the head direction y-axis.'
        },
        'ZAxisMovementDecimalSlider': {
            'level': 2,
            'label': 'Z-Axis Movement',
            'min_value': '-0.90',
            'max_value': '1.20',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the head direction z-axis.'
        },
        'MouthPoutingDecimalSlider': {
            'level': 2,
            'label': 'Mouth Pouting',
            'min_value': '-0.09',
            'max_value': '0.09',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Pouting the mouth.'
        },
        'MouthPursingDecimalSlider': {
            'level': 2,
            'label': 'Mouth Pursing',
            'min_value': '-20.00',
            'max_value': '15.00',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Pursing the mouth.'
        },
        'MouthGrinDecimalSlider': {
            'level': 2,
            'label': 'Mouth Grin',
            'min_value': '0.00',
            'max_value': '15.00',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the mouth grin.'
        },
        'LipsCloseOpenSlider': {
            'level': 2,
            'label': 'Lips Close <--> Open Value',
            'min_value': '-90',
            'max_value': '120',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the closing or opening of the lips.'
        },
        'MouthSmileDecimalSlider': {
            'level': 2,
            'label': 'Mouth Smile',
            'min_value': '-0.30',
            'max_value': '1.30',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the mouth smile.'
        },
        'EyeWinkDecimalSlider': {
            'level': 2,
            'label': 'Eye Wink',
            'min_value': '0.00',
            'max_value': '39.00',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Winking eye.'
        },
        'EyeBrowsDirectionDecimalSlider': {
            'level': 2,
            'label': 'EyeBrows Direction',
            'min_value': '-30.00',
            'max_value': '30.00',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the eyebrows direction.'
        },
        'EyeGazeHorizontalDecimalSlider': {
            'level': 2,
            'label': 'EyeGaze Horizontal',
            'min_value': '-30.00',
            'max_value': '30.00',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the horizontal eyegaze direction.'
        },
        'EyeGazeVerticalDecimalSlider': {
            'level': 2,
            'label': 'EyeGaze Vertical',
            'min_value': '-63.00',
            'max_value': '63.00',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'FaceEditorEnableToggle',
            'requiredToggleValue': True,
            'help': 'Changes the vertical eyegaze direction.'
        },
        'FaceMakeupEnableToggle': {
            'level': 1,
            'label': 'Face Makeup',
            'default': False,
            'help': 'Enable face makeup. Except for hair, eyebrows, eyes and lips.'
        },
        'FaceMakeupRedSlider': {
            'level': 2,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments.'
        },
        'FaceMakeupGreenSlider': {
            'level': 2,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments.'
        },
        'FaceMakeupBlueSlider': {
            'level': 2,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments.'
        },
        'FaceMakeupBlendAmountDecimalSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0.01',
            'max_value': '1.00',
            'default': '0.05',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.00 represents the original color, 1.00 represents the full target color.'
        },
        'HairMakeupEnableToggle': {
            'level': 1,
            'label': 'Hair Makeup',
            'default': False,
            'help': 'Enable hair makeup.'
        },
        'HairMakeupRedSlider': {
            'level': 2,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'HairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments.'
        },
        'HairMakeupGreenSlider': {
            'level': 2,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'HairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments.'
        },
        'HairMakeupBlueSlider': {
            'level': 2,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'HairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments.'
        },
        'HairMakeupBlendAmountDecimalSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0.01',
            'max_value': '1.00',
            'default': '0.05',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'HairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.00 represents the original color, 1.00 represents the full target color.'
        },
        'EyeBrowsMakeupEnableToggle': {
            'level': 1,
            'label': 'EyeBrows Makeup',
            'default': False,
            'help': 'Enable eyebrows makeup.'
        },
        'EyeBrowsMakeupRedSlider': {
            'level': 2,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'EyeBrowsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments.'
        },
        'EyeBrowsMakeupGreenSlider': {
            'level': 2,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'EyeBrowsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments.'
        },
        'EyeBrowsMakeupBlueSlider': {
            'level': 2,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'EyeBrowsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments.'
        },
        'EyeBrowsMakeupBlendAmountDecimalSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0.01',
            'max_value': '1.00',
            'default': '0.05',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'EyeBrowsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.00 represents the original color, 1.00 represents the full target color.'
        },
        'LipsMakeupEnableToggle': {
            'level': 1,
            'label': 'Lips Makeup',
            'default': False,
            'help': 'Enable lips makeup.'
        },
        'LipsMakeupRedSlider': {
            'level': 2,
            'label': 'Red',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'LipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Red color adjustments.'
        },
        'LipsMakeupGreenSlider': {
            'level': 2,
            'label': 'Green',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'LipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Green color adjustments.'
        },
        'LipsMakeupBlueSlider': {
            'level': 2,
            'label': 'Blue',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'LipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blue color adjustments.'
        },
        'LipsMakeupBlendAmountDecimalSlider': {
            'level': 2,
            'label': 'Blend Amount',
            'min_value': '0.01',
            'max_value': '1.00',
            'default': '0.05',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'LipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blend the value: 0.0 represents the original color, 1.0 represents the full target color.'
        },
    },
    'Face Re-Aging': {
        'FaceReagingEnableToggle': {
            'level': 1,
            'label': 'Enable Face Re-Aging',
            'default': False,
            'help': 'Enable the face re-aging filter to adjust the perceived age of the face.'
        },
        'FaceReagingModelSelection': {
            'level': 2,
            'label': 'Re-Aging Model',
            'options': ['Face Aging GAN', 'SAM De-Aging'],
            'default': 'Face Aging GAN',
            'parentToggle': 'FaceReagingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Choose which backend to use. SAM prioritises de-aging while the GAN handles age increases.'
        },
        'FaceReagingCurrentAgeSlider': {
            'level': 3,
            'label': 'Current Age',
            'min_value': '10',
            'max_value': '80',
            'default': '30',
            'step': 1,
            'parentToggle': 'FaceReagingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Approximate age of the person in the source footage.'
        },
        'FaceReagingTargetAgeSlider': {
            'level': 3,
            'label': 'Target Age',
            'min_value': '10',
            'max_value': '80',
            'default': '45',
            'step': 1,
            'parentToggle': 'FaceReagingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Desired age for the re-aged face. The model will age up or down from the current age.'
        },
        'FaceReagingAgeShiftSlider': {
            'level': 3,
            'label': 'Manual Age Offset',
            'min_value': '-50',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceReagingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Optional manual override. Positive values age up and negative values rejuvenate regardless of the age sliders.'
        },
        'FaceReagingStrengthDecimalSlider': {
            'level': 3,
            'label': 'Effect Strength',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.60',
            'step': 0.05,
            'decimals': 2,
            'parentToggle': 'FaceReagingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Controls how strongly the model alters the perceived age.'
        },
        'FaceReagingBlendAmountDecimalSlider': {
            'level': 3,
            'label': 'Blend Amount',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.80',
            'step': 0.05,
            'decimals': 2,
            'parentToggle': 'FaceReagingEnableToggle',
            'requiredToggleValue': True,
            'help': 'Blends between the original face and the re-aged result.'
        },
    }
}
