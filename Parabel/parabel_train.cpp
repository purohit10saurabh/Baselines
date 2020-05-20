#include <iostream>
#include <fstream>
#include <string>
#include "stats.h"
#include <chrono>
#include "parabel.h"

using namespace std;


void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./parabel_train [input model folder name] [start_tree_no] --trn_ft_file [input feature file name] --trn_lbl_file [input label file name] -w [weighted] -k [classifier_kind] -c [classifier_cost] -m [max_leaf] -tcl [classifier_threshold] -tce [centroid_threshold] -ece [clustering_eps] -ecl [classifier_eps] -n [classifier_maxiter]"<<endl;

	cerr<<"-w = param.weighted				: Whether input labels are binary or continuous probability scores, 1=continuous in [0,1], 0=binary. default=0"<<endl;
	cerr<<"-k = param.classifier_kind			: Kind of linear classifier to use. 0=L2R_L2LOSS_SVC_DUAL, 1=L2R_LR_DUAL, 2=L2R_L2LOSS_SVC_PRIMAL (not yet supported), 3=L2R_LR_PRIMAL  (Refer to Liblinear)	default=L2R_L2LOSS_SVC"<<endl;
	cerr<<"-c = param.classifier_cost			: Cost co-efficient for linear classifiers						default=1.0"<<endl;
	cerr<<"-m = param.max_leaf				: Maximum no. of labels in a leaf node. Larger nodes will be split into 2 balanced child nodes.		default=100"<<endl;
	cerr<<"-tcl = param.classifier_threshold			: Threshold value for sparsifying linear classifiers' trained weights to reduce model size.		default=0.1"<<endl;
	cerr<<"-tce = param.centroid_threshold			: Threshold value for sparsifying label centroids to speed up label clustering.		default=0"<<endl;
	cerr<<"-ece = param.clustering_eps			: Eps value for terminating balanced spherical 2-Means clustering algorithm. Algorithm is terminated when successive iterations decrease objective by less than this value.	default=0.0001"<<endl;
	cerr<<"-ecl = param.classifier_eps			: Eps value for logistic regression. default=0.01"<<endl;
	cerr<<"-n = param.classifier_maxiter			: Maximum iterations of algorithm for training linear classifiers			default=20"<<endl;
	cerr<<"The feature and label input files are expected to be in sparse matrix text format. Refer to README.txt for more details."<<endl;

	exit(1);
}

std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	string model_folder = string( argv[1] );
	check_valid_foldername( model_folder );

	_int start_tree_no = stoi( argv[2] );

	string param_file_name = model_folder + "/Params." + to_string( start_tree_no ) + ".txt";
	check_valid_filename( param_file_name, false );

	string trn_ft_file;
	string trn_lbl_file;
	for( _int i=0; i<argc-3; i+=2 )
	{
		string opt = string(argv[3+i]);
		string sval = string(argv[3+i+1]);

		if( opt=="--trn_ft_file" )
			trn_ft_file = sval;	
		else if( opt=="--trn_lbl_file" )
			trn_lbl_file = sval;
	}

	cout << "loading the datasets " << endl;

	SMatF* trn_X_Xf = new SMatF(trn_ft_file);
	SMatF* trn_X_Y = new SMatF(trn_lbl_file);

	cout << "The shape of trn_X_Xf is " << trn_X_Xf->nc << " " << trn_X_Xf->nr << endl;
	cout << "The shape of trn_X_Y is " << trn_X_Y->nc << " " << trn_X_Y->nr << endl;

	Param param;
	param.parse( argc-3, argv+3 );
	param.num_Xf = trn_X_Xf->nr;
	param.num_Y = trn_X_Y->nr;
	param.write( param_file_name );

	param.num_trn = trn_X_Xf->nc;
	
	trn_X_Xf->unit_normalize_columns();
	trn_X_Xf->append_bias_feat(param.bias_feat);

	SMatF* trn_Y_X = trn_X_Y->transpose();

	_float model_size = 0;

	auto start_time = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < param.num_tree; ++i)
	{
		string tree_file_name = model_folder + "/Tree." + to_string( start_tree_no + i ) + ".bin";
		check_valid_filename( tree_file_name, false );

		ofstream fout;
		fout.open(tree_file_name, ios::out | ios::binary);

		ParabelTrain parabel_train(trn_X_Xf, trn_Y_X, param, start_tree_no + i, fout);
		parabel_train.train_tree();

		model_size += filesize(tree_file_name.c_str());
	}
	if(param.tail_classifier)
	{
		_float tc_train_time;
		string model_file_name = model_folder + "/tail_classifier_model.bin";
		SMatF* model_mat = tail_classifier_train( trn_X_Xf, trn_X_Y, tc_train_time );

		ofstream fout(model_file_name, ios::out | ios::binary);
		model_mat->writeBin(fout);

		delete model_mat;
		model_size += filesize(model_file_name.c_str());
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	auto elasped_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
	cout << "model size is : " << model_size / pow(2, 20) << " MB" << endl;
	cout << "train time is : " << elasped_time << " sec." << endl;

	delete trn_Y_X;
	delete trn_X_Xf;
	delete trn_X_Y;
}
