experiment_path = 'experiments/chosing_one_tag'
experiment_name = 'exploring_multiframe_v8'

settings = {
    'experiment_name' : experiment_name,
    'experiment_path' : experiment_path,

    'dataset_filename': '/home/ubuntu/datasets_vilynx/labels/multi_frame_new_tags_images_mult8.csv',
    'images_path': '/home/ubuntu/datasets_vilynx/uploaded_images/images_mult8',
    'minimum_samples_per_tag': 40,

    'output_file_train' : '%s/data/files/train.txt'  % experiment_path, ## DO NOT MODIFY THIS ONE
    'output_file_test' : '%s/data/files/val.txt'  % experiment_path, ## DO NOT MODIFY THIS ONE
    'processed_labels_csv': '%s/data/files/%s_prepared_dataset.csv'  % (experiment_path, experiment_name), ## DO NOT MODIFY THIS ONE
    'processed_labels_2_original_label' : '%s/data/files/%s_classes.npy'  % (experiment_path, experiment_name), ## DO NOT MODIFY THIS ONE

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


