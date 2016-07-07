
experiment_path = 'experiments/processing_videos'
experiment_name = 'testing_refactoring'
settings = {
    'experiment_name' : experiment_name,
    'experiment_path' : experiment_path,

    'dataset_filename': '/home/ubuntu/datasets_vilynx/labels/multi_frame_new_tags.csv',
    #'images_path': '/home/ubuntu/uploaded_images/images',
    'images_path': '/home/ubuntu/victor_tests/vilynx_bitbucket/vilynx-dl2/data/images',
    'minimum_samples_per_tag': 30,
    'train_size': 0.75,
    'frames_per_video': 60,
    'batch_size':240,

    'path_for_files' : '%s/data/files'  % experiment_path, ## DO NOT MODIFY THIS ONE
    'output_file_train' : '%s_train.txt' % experiment_name,
    'output_file_test' : '%s_val.txt' % experiment_name, ## DO NOT MODIFY THIS ONE
    'dict_dateset': '%s_dict_dataset.csv' % experiment_name, ## DO NOT MODIFY THIS ONE
    'processed_labels_2_original_label' : '%s_classes.npy' % experiment_name, ## DO NOT MODIFY THIS ONE

    'map_template2file': {
    	'TEST': {
		            'inputprototxt' :                   '%s/base_network/my_network/base_files/input_layers_base/deploy_input_layers_base.prototxt' % experiment_path,
		            'evaltrainstage' :                  '%s/base_network/my_network/base_files/output_layers_templates/empty_template.prototxt' % experiment_path,
		            'crossentropylossintermediate' :    '%s/base_network/my_network/base_files/output_layers_templates/empty_template.prototxt' % experiment_path
		        },
        'TRAIN': {
        			'inputprototxt' : 					'%s/base_network/my_network/base_files/input_layers_base/train_layers_base.prototxt' % experiment_path,
			        'evaltrainstage' : 					'%s/base_network/my_network/base_files/output_layers_templates/final_output_base.prototxt' % experiment_path,
			        'crossentropylossintermediate' : 	'%s/base_network/my_network/base_files/output_layers_templates/crossentropylossintermediate.prototxt' % experiment_path,
        		}

    },

    'PROTOTXT_BASE':'%s/base_network/my_network/base_files/googlenetbase.prototxt' % experiment_path,
    'SOLVER_BASE':'%s/base_network/my_network/base_files/quick_solver_base.prototxt' % experiment_path,

    #Path to the folder where we can wait the initial weights in case we want to use some
    'INITIAL_WEIGHTS':'/home/ubuntu/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
    'PROTOTXT_READY_PATH': '%s/data/base_network/my_network/ready_files' % experiment_path,
    'PROTOTXT_READY':'%s/data/base_network/my_network/ready_files/%s_ready_network.prototxt' % (experiment_path, experiment_name),
    'SOLVER_READY':'%s/data/base_network/my_network/ready_files/%s_ready_solver.prototxt' % (experiment_path, experiment_name),

    #Path where we will save the snapshots which are going to be produced when training the network
    'SNAPSHOT_PREFIX':'%s/data/snapshots' % experiment_path,
    'LOGS_PATH':'%s/data/logs' % experiment_path
}


