#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "common_functions.h"

using namespace std;


int main(int argc, char* argv[])
{
	string score_file = string( argv[1] );
	check_valid_filename( score_file, true );
	
	string lbl_file = string( argv[2] );
	check_valid_filename( lbl_file, true );
	
	int K = (int) stoi(string(argv[3]));

	_bool input_format_is_binary = (_bool)(stoi(argv[4]));
	
	SMatF* score_mat = new SMatF(score_file, input_format_is_binary );
	cout<<"score file read "<<score_file<<endl;
	
	SMatF* lbl_mat = new SMatF(lbl_file, input_format_is_binary);
	cout<<"lbl file read "<<lbl_file<<endl;

	cout<<"num_inst="<<score_mat->nc<<" num_lbl="<<lbl_mat->nr<<endl;
	assert(score_mat->nc==lbl_mat->nc);
	assert(score_mat->nr==lbl_mat->nr);
	top_k(score_mat, lbl_mat, K);
	
	delete score_mat;
	delete lbl_mat;
}
