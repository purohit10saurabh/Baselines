#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>
#include <chrono>
#include "parabel.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./parabel_predict [input model folder name] [tree_no] [output score file name] --tst_ft_file [input feature file name] -B 10 -p 0 -pf 10.0 -ps -0.05 -C -1"<<endl<<endl;
	cerr<<"-B = param.beam_width				: Beam search width for fast, approximate prediction					default=10"<<endl;
	cerr<<"-p = param.per_label_predict	: Predict top test points per each label. Useful in DSA-like scenarios 0=predict top labels per point, 1=predict top points per label default=[value saved in trained model]"<<endl;
	cerr<<"-pf = param.per_label_predict_factor : per_label_predict_factor*max_leaf' number of test points are finally passed down to each leaf node. default=10.0"<<endl;
	cerr<<"-ps = param.per_label_predict_slope : slope of the linear function which decides how many test points are passed from parent to child. Function is linear in node depth. default=-0.05"<<endl;
	cerr<<"-C = param.prediction_chunk_size			: Partition the large test datasets into chunks for prediction under limited memory. If chunk size is -1, no dataset chunking is done. default=[as saved in model, -1]"<<endl;
	cerr<<"-s = param.start_tree		: Tree no.		default=0"<<endl;

	cerr<<"The feature and score files are expected to be in sparse matrix text format. Refer to README.txt for more details"<<endl;
	exit(1);
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	string model_folder = string( argv[1] );
	check_valid_foldername( model_folder );

	_int start_tree_no = stoi( argv[2] );

	string param_file_name = model_folder + "/Params." + to_string( start_tree_no ) + ".txt";
	check_valid_filename( param_file_name, true );

	Param param( param_file_name );
	param.parse( argc-4, argv+4 );

	string tst_ft_file;
	for( _int i=0; i<argc-4; i+=2 )
	{
		string opt = string(argv[4+i]);
		string sval = string(argv[4+i+1]);

		if( opt=="--tst_ft_file" )
			tst_ft_file = sval;	
	}

	cout << "Loading all mats... " << endl;

	vector<string> ids;  // store the point ids as strings
	SMatF* tst_X_Xf = new SMatF(tst_ft_file);

	cout << "loaded the test mat " << endl;

	tst_X_Xf->unit_normalize_columns();
	tst_X_Xf->append_bias_feat(param.bias_feat);

	cout << "Shape of tst_X_Xf " << tst_X_Xf->nc << " " << tst_X_Xf->nr << endl;
	_int num_samples = tst_X_Xf->nr;
	auto start_time = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < param.num_tree; ++i)
	{
		string tree_file_name = model_folder + "/Tree." + to_string( start_tree_no + i ) + ".bin";
		check_valid_filename( tree_file_name, true );

		string temp_score_file_name = string( argv[3] ) + to_string( start_tree_no + i );
		check_valid_filename( temp_score_file_name, false );

		Tree* tree = new Tree( tree_file_name );
		cout << "Total nodes in the tree are " << tree->nodes.size() << endl;

		ofstream fout(temp_score_file_name);

		// if(param.per_label_predict)
		// {
		// 	predict_tree_per_label(tst_X_Xf, tree, param, fout);
		// }
		// else
		// {
		// 	predict_tree(tst_X_Xf, tree, param, fout);	
		// 	delete tree;		
		// }

		ParabelPredict parabel_predict(tst_X_Xf, tree, param, fout);

		if(param.per_label_predict)
			parabel_predict.predict_tree_per_label();
		else
			parabel_predict.predict_tree();

		delete tree;
	}
	
	string score_file_name = string( argv[3] );
	check_valid_filename( score_file_name, false );

	_int num_X = tst_X_Xf->nc;
	_int num_Y = param.num_Y;
	SMatF* score_mat = new SMatF( num_Y, num_X );

	for(int i = 0; i < param.num_tree; ++i)
	{
		cout << "Taking ensemble of predictions..." << endl;

		string temp_score_file_name = string( argv[3] ) + to_string( start_tree_no + i );
		ifstream fin(temp_score_file_name);

		SMatF* tree_score_mat = new SMatF( num_Y, num_X );

		string line;
		while(getline(fin, line))
		{
			std::istringstream iss(line);

			char colon;
			_int col_no; iss >> col_no;

			_int label_id;
			_float label_score;

			vector<pair<_int, _float>> scores;

			while(iss >> label_id >> colon >> label_score)
				scores.push_back(pair<_int, _float>(label_id, label_score));

			sort(scores.begin(), scores.end());

			tree_score_mat->size[col_no] = scores.size();
			tree_score_mat->data[col_no] = new pairIF[ scores.size() ];

			for(int j = 0; j < scores.size(); ++j)
				tree_score_mat->data[col_no][j] = scores[j];
		}

		score_mat->add( tree_score_mat );
		delete tree_score_mat;
		remove(temp_score_file_name.c_str());
	}

	for(_int i=0; i<score_mat->nc; i++)
		for(_int j=0; j<score_mat->size[i]; j++)
			score_mat->data[i][j].second /= param.num_tree;
	auto end_time = std::chrono::high_resolution_clock::now();
	float elasped_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	ofstream fout( score_file_name );
	fout << (*score_mat);

	fout.close();
	cout << "prediction time (per sample) is : " << elasped_time/num_samples << " msec." << endl;	
	delete tst_X_Xf;
	delete score_mat;
}