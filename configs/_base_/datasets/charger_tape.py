dataset_info = dict(
    dataset_name='charger_tape',
    paper_info = '',
    keypoint_info={
        0:
        dict(name='roof', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='vert_pole',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='hor_pole_right',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='hor_pole_left',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('roof', 'vert_pole'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('vert_pole', 'hor_pole_right'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('hor_pole_right', 'hor_pole_left'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('hor_pole_left', 'roof'), id=3, color=[255, 128, 0]),
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.107, 0.107, 0.107, 0.107
    ])
