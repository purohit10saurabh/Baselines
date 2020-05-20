#include <iostream>
#include <string>
#include <omp.h> 
#include <cmath>
#include <map>
#include "config.h"
#include "utils.h"  
#include "mat.h" 
using namespace std;
void top_k(SMatF* score_mat, SMatF* lbl_mat, int K)
{
	_int num_inst = score_mat->nc;
	_int num_lbl = lbl_mat->nr;
	_float avg_pred = 0;
	_float avg_lbl = 0;

	int* p = new int[K];
	for(int i=0;i<K;i++)
		p[i]=0;

	for(_int i=0;i<num_inst;i++)
	{
		avg_pred += score_mat->size[i];
		avg_lbl += lbl_mat->size[i];

		std::map<_int, _int> lbl_map;
		for(_int j=0;j<lbl_mat->size[i];j++)
			lbl_map[lbl_mat->data[i][j].first] = 1;
		int temp = 0;
		for(_int j=0;j<score_mat->size[i];j++)
		{
			if(lbl_map.count(score_mat->data[i][j].first)>0)
				temp++;
		}
		int k = temp;
		if (k>K)
			k = K;
		for(_int j=1;j<=K;j++)
		{
			if(j>temp)
				p[j-1] += temp;
			else
				p[j-1] += j;
		}
	}
	printf("Average labels per point = %f\n",avg_lbl/(float)num_inst);
	printf("Average predictions per point = %f\n",avg_pred/(float)num_inst);
	for(int i=0;i<K;i++)
		printf("Top%d = %f\n",i+1,(float)p[i]/((float)(num_inst*(i+1))));
}

void precision_k(SMatF* score_mat, SMatF* lbl_mat, int K)
{
	_int num_inst = score_mat->nc;
	_int num_lbl = lbl_mat->nr;
	int* p = new int[K];
	for(int i=0;i<K;i++)
		p[i]=0;
	for(_int i=0;i<num_inst;i++)
	{
		std::map<_int, _int> lbl_map;
		for(_int j=0;j<lbl_mat->size[i];j++)
			lbl_map[lbl_mat->data[i][j].first] = 1;
			
		pairIF* vec = score_mat->data[i];
		sort(vec, vec+score_mat->size[i], comp_pair_by_second_desc<_int,_float>);
		_int k = K;
		if (k>score_mat->size[i])
			k = score_mat->size[i];
		for(_int j=0;j<k;j++)
		{
			if(lbl_map.count(vec[j].first)>0)
			{
				for(_int pos=j;pos<K;pos++)
					p[pos]++;
			}
		}
	}
	for(int i=0;i<K;i++)
		printf("Precision@%d = %f\n",i+1,(float)p[i]/((float)(num_inst*(i+1))));
}
void ndcg_k(SMatF* score_mat, SMatF* lbl_mat, int K)
{
	_int num_inst = score_mat->nc;
	_int num_lbl = lbl_mat->nr;

	float* ndcg = new float[K];
	for(int i=0;i<K;i++)
		ndcg[i]=0;
	for(_int i=0;i<num_inst;i++)
	{
		if (lbl_mat->size[i]==0)
			continue;

		std::map<_int, _int> lbl_map;
		float* den = new float[K];
		float* n = new float[K];
		for(int j=0;j<K;j++)
		{
			den[j]=0;
			n[j]=0.0;
		}
		for(_int j=0;j<lbl_mat->size[i];j++)
		{
			if(j==0)
				den[j] = 1;
			else if(j<K && j>0)
				den[j] = den[j-1] + log(2.0)/log(2+j);
			lbl_map[lbl_mat->data[i][j].first] = 1;
		}
		for(_int j=lbl_mat->size[i];j<K;j++)
			den[j] = den[lbl_mat->size[i]-1];
		pairIF* vec = score_mat->data[i];
		sort(vec, vec+score_mat->size[i], comp_pair_by_second_desc<_int,_float>);

		_int k = K;
		if (k>score_mat->size[i])
			k = score_mat->size[i];

		for(_int j=0;j<k;j++)
		{
			if(lbl_map.count(vec[j].first)>0)
			{
				for(_int pos=j;pos<K;pos++)
					n[pos] += log(2.0)/log(2+j);
			}
		}
		for(_int j=0; j<K; j++)
			ndcg[j] += n[j]/den[j];

	}
	for(int i=0;i<K;i++)
		printf("nDCG@%d = %f\n",i+1,ndcg[i]/((float)(num_inst)));
}
inline void concatenate_IMat_files(string outfile, string* fname, _int num_files, _int nc, _int nr, _bool input_format_is_binary, _bool output_format_is_binary)
{
	ofstream fout;
	ifstream* fin = new ifstream[num_files];
	_int* offset = new _int[num_files+1];
	offset[0] = 0;
	vector<_int> inds;
	_int ind = 0;
	_int col_size = 0;
	_int siz = 0;
	if(input_format_is_binary)
	{
		_int tnc,tnr;	
		for(_int file_no=0;file_no<num_files;file_no++)
		{
			fin[file_no].open(fname[file_no], std::ios::binary);
			fin[file_no].read((char *)(&(tnc)), sizeof(_int));
			fin[file_no].read((char *)(&(tnr)), sizeof(_int));
			offset[file_no+1] = offset[file_no]+tnr;
		}
		if(output_format_is_binary)
		{
			fout.open(outfile, std::ios::binary);
			fout.write((char *)(&(nc)), sizeof(_int));
			fout.write((char *)(&(nr)), sizeof(_int));
		}
		else
		{
			fout.open(outfile);
			fout<<nc<<" "<<nr<<endl;
		}
		for(_int column = 0; column < (nc); ++column)
		{
			if(column%100000==0)
				printf("%d\n",column);
			inds.clear();
			for(_int file_no=0;file_no<num_files;file_no++)
			{
				fin[file_no].read((char *)(&col_size), sizeof(_int));
				for (_int row = 0; row < (col_size); ++row)
				{
					fin[file_no].read((char *)(&ind), sizeof(_int));
					inds.push_back(ind + offset[file_no]);
				}
			}
			assert(inds.size()==0 || inds[inds.size()-1]<nr);
			if(output_format_is_binary)
			{
				siz = inds.size();
				fout.write((char *)(&(siz)), sizeof(_int));
				for(_int row=0; row<inds.size(); row++)
					fout.write((char *)(&inds[row]), sizeof(_int));
			}
			else
			{
				for(_int row=0; row<inds.size(); row++)
				{
					if(row==0)
						fout<<inds[row];
					else
						fout<<" "<<inds[row];
				}
				fout<<endl;	
			}
		}
		fout.close();
	}
	else
	{
		_int tnc,tnr;	
		for(_int file_no=0;file_no<num_files;file_no++)
		{
			fin[file_no].open(fname[file_no]);
			fin[file_no]>>tnc>>tnr;
			offset[file_no+1] = offset[file_no]+tnr;
			fin[file_no].ignore();
		}
		if(output_format_is_binary)
		{
			fout.open(outfile, std::ios::binary);
			fout.write((char *)(&(nc)), sizeof(_int));
			fout.write((char *)(&(nr)), sizeof(_int));
		}
		else
		{
			fout.open(outfile);
			fout<<nc<<" "<<nr<<endl;
		}
		for(_int column = 0; column < (nc); ++column)
		{
			if(column%100000==0)
				printf("%d\n",column);
			inds.clear();
			string line;
			_int pos = 0;
			_int next_pos;
			for(_int file_no=0;file_no<num_files;file_no++)
			{
				pos = 0;
				getline(fin[file_no],line);
				line += "\n";
				while(next_pos=line.find_first_of(" \n",pos))
				{
					if((size_t)next_pos==string::npos)
						break;
					inds.push_back(stoi(line.substr(pos,next_pos-pos))+offset[file_no]);
					pos = next_pos+1;
				}
				assert(inds[inds.size()-1]<nr);
			}
			if(output_format_is_binary)
			{
				siz = inds.size();
				fout.write((char *)(&(siz)), sizeof(_int));
				for(_int row=0; row<inds.size(); row++)
					fout.write((char *)(&inds[row]), sizeof(_int));
			}
			else
			{
				for(_int row=0; row<inds.size(); row++)
				{
					if(row==0)
						fout<<inds[row];
					else
						fout<<" "<<inds[row];
				}
				fout<<endl;	
			}
		}
		fout.close();
	}
	for(_int file_no=0;file_no<num_files;file_no++)
		fin[file_no].close();
}

inline void concatenate_bin_IMat_files_and_split_equally(string* outfiles, string* fname, _int num_files, _int nc0, _int nr0)
{
	ifstream* fin = new ifstream[num_files];
	_int* offset = new _int[num_files+1];
	offset[0] = 0;
	_int tnc,tnr;	
	
	for(_int file_no=0;file_no<num_files;file_no++)
	{
		fin[file_no].open(fname[file_no], std::ios::binary);
		fin[file_no].read((char *)(&(tnc)), sizeof(_int));
		fin[file_no].read((char *)(&(tnr)), sizeof(_int));
		offset[file_no+1] = offset[file_no]+tnr;
	}	
	
	_int ctr = 0;	
	_int nc = ceil((_float)nc0/(_float)num_files)+1;
	
	ofstream* fout = new ofstream[num_files]();
	for(_int i=0;i<num_files;i++)
	{
		fout[i].open(outfiles[i], std::ios::binary);
		tnc = nc0-ctr;
		if (tnc>nc)
		{
			fout[i].write((char *)(&(nc)), sizeof(_int));
			fout[i].write((char *)(&(nr0)), sizeof(_int));
			printf("File#%d:nc = %d\tnr=%d\n",i,nc,nr0);
		}
		else
		{
			fout[i].write((char *)(&(tnc)), sizeof(_int));
			fout[i].write((char *)(&(nr0)), sizeof(_int));
			printf("File#%d:nc = %d\tnr=%d\n",i,tnc,nr0);
		}
		ctr = ctr+nc;
	}
	
	vector<_int> inds;
	_int ind = 0;
	_int col_size = 0;
	_int siz = 0;
	_int out_file_no = -1;
	for(_int column = 0; column < (nc0); ++column)
	{
		if(column%100000==0)
			printf("%d\n",column);
		if ((column%nc) == 0)
			out_file_no++;
		inds.clear();
		for(_int file_no=0;file_no<num_files;file_no++)
		{
			fin[file_no].read((char *)(&col_size), sizeof(_int));
			for (_int row = 0; row < (col_size); ++row)
			{
				fin[file_no].read((char *)(&ind), sizeof(_int));
				inds.push_back(ind + offset[file_no]);
			}
		}
		assert(inds.size()==0 || inds[inds.size()-1]<nr0);
		siz = inds.size();
		fout[out_file_no].write((char *)(&(siz)), sizeof(_int));
		for(_int row=0; row<inds.size(); row++)
			fout[out_file_no].write((char *)(&inds[row]), sizeof(_int));
			
	}
	for(_int file_no=0;file_no<num_files;file_no++)
	{
		fin[file_no].close();
		fout[file_no].close();
	}
}

inline void merge_smat_files(string outfile, string* fname, _int num_files, _int nc, _int nr, _bool input_format_is_binary)
{
	if(input_format_is_binary)
	{
		ofstream fout;
		fout.open(outfile, std::ios::binary);
		fout.write((char *)(&(nc)), sizeof(_int));
		fout.write((char *)(&(nr)), sizeof(_int));
		printf("nc = %d nr = %d\n",nc,nr);
		ifstream fin;
		_int col_size = 0;
		_int siz = 0;
		char* memblock;
		for(_int i=0;i<num_files;i++)
		{
			printf("Writing file# %d\n",i);
			fin.open(fname[i], std::ios::binary);
			fin.read((char *)(&(nc)), sizeof(_int));
			fin.read((char *)(&(nr)), sizeof(_int));
			for(_int j=0; j<nc; j++)
			{
				fin.read((char *)(&(col_size)), sizeof(_int));
				siz = col_size*(sizeof(_int)+sizeof(_float));
				memblock = new char [siz];
				fin.read(memblock, siz);
				fout.write((char *)(&(col_size)), sizeof(_int));
				fout.write(memblock, siz);
			}
			fin.close();
		}
		fout.close();
	}
	else
	{
		ofstream fout;
		fout.open(outfile);
		fout<<nc<<" "<<nr<<endl;
		printf("nc = %d nr = %d\n",nc,nr);
		ifstream fin;
		for(_int i=0;i<num_files;i++)
		{
			printf("Writing file# %d\n",i);
			fin.open(fname[i]);
			fin>>nc>>nr;
			fin.ignore();
			for(_int j=0; j<nc; j++)
			{
				string line;
				getline(fin,line);
				fout<<line<<endl;
			}
			fin.close();
		}
		fout.close();
	}
}

inline void merge_dmat_files(string outfile, string* fname, _int num_files, _int nc, _int nr, _bool input_format_is_binary)
{
	if(input_format_is_binary)
	{
		ofstream fout;
		fout.open(outfile, std::ios::binary);
		fout.write((char *)(&(nc)), sizeof(_int));
		fout.write((char *)(&(nr)), sizeof(_int));
		printf("nc = %d nr = %d\n",nc,nr);
		ifstream fin;
		_int col_size = 0;
		long long int siz = 0;
		char* memblock;
		for(_int i=0;i<num_files;i++)
		{
			printf("Writing file# %d\n",i);
			fin.open(fname[i], std::ios::binary);
			fin.read((char *)(&(nc)), sizeof(_int));
			fin.read((char *)(&(nr)), sizeof(_int));
			for(_int j=0; j<nc; j++)
			{
				siz = nr*(sizeof(_float));
				memblock = new char [siz];
				fin.read(memblock, siz);
				fout.write(memblock, siz);
			}
			fin.close();
		}
		fout.close();
	}
	else
	{
		ofstream fout;
		fout.open(outfile);
		fout<<nc<<" "<<nr<<endl;
		printf("nc = %d nr = %d\n",nc,nr);
		ifstream fin;
		for(_int i=0;i<num_files;i++)
		{
			printf("Writing file# %d\n",i);
			fin.open(fname[i]);
			fin>>nc>>nr;
			fin.ignore();
			for(_int j=0; j<nc; j++)
			{
				string line;
				getline(fin,line);
				fout<<line<<endl;
			}
			fin.close();
		}
		fout.close();
	}
}

inline void split_bin_IMat_file_equally(string infile, string* fname, _int num_files)
{
	_int nc0, nr0;
	ifstream fin;
	fin.open(infile, std::ios::binary);
	fin.read((char *)(&(nc0)), sizeof(_int));
	fin.read((char *)(&(nr0)), sizeof(_int));
	printf("nc = %d\tnr=%d\n",nc0,nr0);
	
	_int nc = ceil((_float)nc0/(_float)num_files)+1;
	_int ctr = 0;
	ofstream* fout = new ofstream[num_files]();
	for(_int i=0;i<num_files;i++)
	{
		fout[i].open(fname[i], std::ios::binary);
		if ((nc0-ctr)>nc)
		{
			fout[i].write((char *)(&(nc)), sizeof(_int));
			fout[i].write((char *)(&(nr0)), sizeof(_int));
			printf("File#%d:nc = %d\tnr=%d\n",i,nc,nr0);
		}
		else
		{
			_int tnc = nc0-ctr;
			fout[i].write((char *)(&(tnc)), sizeof(_int));
			fout[i].write((char *)(&(nr0)), sizeof(_int));
			printf("File#%d:nc = %d\tnr=%d\n",i,tnc,nr0);
		}
		ctr = ctr+nc;
	}
	
	_int file_no = -1;
	_int col_size = 0;
	char* memblock;
	ctr = 0;
	for(_int i=0; i<nc0; i++)
	{
		if ((i%100000)==0)
			printf("%d\n",i);
		if ((i%nc) == 0)
			file_no++;
		fin.read((char *)(&(col_size)), sizeof(_int));
		memblock = new char [col_size*sizeof(_int)];
		fin.read(memblock, col_size*sizeof(_int));
		fout[file_no].write((char *)(&(col_size)), sizeof(_int));
		fout[file_no].write(memblock, col_size*sizeof(_int));
	}
	fin.close();
	for(_int i =0;i<num_files;i++)
		fout[i].close();
}
inline void split_dmat_as_per_split_file(string infile, string split_file, string* fname, _int num_files, _bool input_format_is_binary)
{
	if(input_format_is_binary)
	{
		_int nc0, nr0, nc_split;
		ifstream fsplit;
		vector<_int> nc(num_files,0);
		fsplit.open(split_file);
		fsplit>>nc_split;
		_int s;
		for(_int i=0;i<nc_split;i++)
		{
			fsplit>>s;
			nc[s]++;
		}
		fsplit.close();
		
		ifstream fin;
		fin.open(infile, std::ios::binary);
		fin.read((char *)(&(nc0)), sizeof(_int));
		fin.read((char *)(&(nr0)), sizeof(_int));
		printf("nc = %d\tnr=%d\n",nc0,nr0);
		assert(nc0==nc_split);
		
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i], std::ios::binary);
			fout[i].write((char *)(&(nc[i])), sizeof(_int));
			fout[i].write((char *)(&(nr0)), sizeof(_int));
			printf("FILE#%d nc = %d\tnr=%d\n",i,nc[i],nr0);
		}
	
		fsplit.open(split_file);
		fsplit>>nc_split;
		
		_int file_no;
		_int siz = 0;
		char* memblock;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			fsplit>>file_no;
			siz = nr0*(sizeof(_float));
			memblock = new char [siz];
			fin.read(memblock, siz);
			fout[file_no].write(memblock, siz);
		}
		fin.close();
		fsplit.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
	else
	{
		_int nc0, nr0, nc_split;
		ifstream fsplit;
		vector<_int> nc(num_files,0);
		fsplit.open(split_file);
		fsplit>>nc_split;
		_int s;
		for(_int i=0;i<nc_split;i++)
		{
			fsplit>>s;
			nc[s]++;
		}
		fsplit.close();
		
		ifstream fin;
		fin.open(infile);
		fin>>nc0>>nr0;
		printf("nc = %d\tnr=%d\n",nc0,nr0);
		assert(nc0==nc_split);
	
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i]);
			fout[i]<<nc[i]<<" "<<nr0<<endl;
			printf("FILE#%d nc = %d\tnr=%d\n",i,nc[i],nr0);
		}
		
		fsplit.open(split_file);
		fsplit>>nc_split;
		
		fin.ignore();
		_int file_no;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			fsplit>>file_no;
			string line;
			getline(fin,line);
			fout[file_no]<<line<<endl;
		}
		fin.close();
		fsplit.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
}

inline void split_dmat_file_equally(string infile, string* fname, _int num_files, _bool input_format_is_binary)
{
	if(input_format_is_binary)
	{
		_int nc0, nr0;
		ifstream fin;
		fin.open(infile, std::ios::binary);
		fin.read((char *)(&(nc0)), sizeof(_int));
		fin.read((char *)(&(nr0)), sizeof(_int));
		printf("nc = %d\tnr=%d\n",nc0,nr0);
	
		_int nc = ceil((_float)nc0/(_float)num_files)+1;
		_int ctr = 0;
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i], std::ios::binary);
			if ((nc0-ctr)>nc)
			{
				fout[i].write((char *)(&(nc)), sizeof(_int));
				fout[i].write((char *)(&(nr0)), sizeof(_int));
			}
			else
			{
				_int tnc = nc0-ctr;
				fout[i].write((char *)(&(tnc)), sizeof(_int));
				fout[i].write((char *)(&(nr0)), sizeof(_int));
			}
			ctr = ctr+nc;
		}
	
		_int file_no = -1;
		_int siz = 0;
		char* memblock;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			if ((i%nc) == 0)
				file_no++;
			siz = nr0*(sizeof(_float));
			memblock = new char [siz];
			fin.read(memblock, siz);
			fout[file_no].write(memblock, siz);
		}
		fin.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
	else
	{
		_int nc0, nr0;
		ifstream fin;
		fin.open(infile);
		fin>>nc0>>nr0;
		printf("nc = %d\tnr=%d\n",nc0,nr0);
	
		_int nc = ceil((_float)nc0/(_float)num_files)+1;
		_int ctr = 0;
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i]);
			if ((nc0-ctr)>nc)
				fout[i]<<nc<<" "<<nr0<<endl;
			else
				fout[i]<<nc0-ctr<<" "<<nr0<<endl;
			ctr = ctr+nc;
		}
		
		fin.ignore();
		_int file_no = -1;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			if ((i%nc) == 0)
				file_no++;
			
			string line;
			getline(fin,line);
		fout[file_no]<<line<<endl;
		}
		fin.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
}
inline void split_smat_as_per_split_file(string infile, string split_file, string* fname, _int num_files, _bool input_format_is_binary)
{
	if(input_format_is_binary)
	{
		_int nc0, nr0, nc_split;
		ifstream fsplit;
		vector<_int> nc(num_files,0);
		fsplit.open(split_file);
		fsplit>>nc_split;
		_int s;
		for(_int i=0;i<nc_split;i++)
		{
			fsplit>>s;
			nc[s]++;
		}
		fsplit.close();
		
		ifstream fin;
		fin.open(infile, std::ios::binary);
		fin.read((char *)(&(nc0)), sizeof(_int));
		fin.read((char *)(&(nr0)), sizeof(_int));
		printf("nc = %d\tnr=%d\n",nc0,nr0);
		assert(nc0==nc_split);
		
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i], std::ios::binary);
			fout[i].write((char *)(&(nc[i])), sizeof(_int));
			fout[i].write((char *)(&(nr0)), sizeof(_int));
			printf("FILE#%d nc = %d\tnr=%d\n",i,nc[i],nr0);
		}
	
		fsplit.open(split_file);
		fsplit>>nc_split;
		
		_int file_no;
		_int col_size = 0;
		_int siz = 0;
		char* memblock;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			fsplit>>file_no;
			fin.read((char *)(&(col_size)), sizeof(_int));
			siz = col_size*(sizeof(_int)+sizeof(_float));
			memblock = new char [siz];
			fin.read(memblock, siz);
			fout[file_no].write((char *)(&(col_size)), sizeof(_int));
			fout[file_no].write(memblock, siz);
		}
		fin.close();
		fsplit.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
	else
	{
		_int nc0, nr0, nc_split;
		ifstream fsplit;
		vector<_int> nc(num_files,0);
		fsplit.open(split_file);
		fsplit>>nc_split;
		_int s;
		for(_int i=0;i<nc_split;i++)
		{
			fsplit>>s;
			nc[s]++;
		}
		fsplit.close();
		
		ifstream fin;
		fin.open(infile);
		fin>>nc0>>nr0;
		printf("nc = %d\tnr=%d\n",nc0,nr0);
		assert(nc0==nc_split);
	
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i]);
			fout[i]<<nc[i]<<" "<<nr0<<endl;
			printf("FILE#%d nc = %d\tnr=%d\n",i,nc[i],nr0);
		}
		
		fsplit.open(split_file);
		fsplit>>nc_split;
		
		fin.ignore();
		_int file_no;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			fsplit>>file_no;
			string line;
			getline(fin,line);
			fout[file_no]<<line<<endl;
		}
		fin.close();
		fsplit.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
}

inline void split_smat_file_equally(string infile, string* fname, _int num_files, _bool input_format_is_binary)
{
	if(input_format_is_binary)
	{
		_int nc0, nr0;
		ifstream fin;
		fin.open(infile, std::ios::binary);
		fin.read((char *)(&(nc0)), sizeof(_int));
		fin.read((char *)(&(nr0)), sizeof(_int));
		printf("nc = %d\tnr=%d\n",nc0,nr0);
	
		_int nc = ceil((_float)nc0/(_float)num_files)+1;
		_int ctr = 0;
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i], std::ios::binary);
			if ((nc0-ctr)>nc)
			{
				fout[i].write((char *)(&(nc)), sizeof(_int));
				fout[i].write((char *)(&(nr0)), sizeof(_int));
			}
			else
			{
				_int tnc = nc0-ctr;
				fout[i].write((char *)(&(tnc)), sizeof(_int));
				fout[i].write((char *)(&(nr0)), sizeof(_int));
			}
			ctr = ctr+nc;
		}
	
		_int file_no = -1;
		_int col_size = 0;
		_int siz = 0;
		char* memblock;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			if ((i%nc) == 0)
				file_no++;
			fin.read((char *)(&(col_size)), sizeof(_int));
			siz = col_size*(sizeof(_int)+sizeof(_float));
			memblock = new char [siz];
			fin.read(memblock, siz);
			fout[file_no].write((char *)(&(col_size)), sizeof(_int));
			fout[file_no].write(memblock, siz);
		}
		fin.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
	else
	{
		_int nc0, nr0;
		ifstream fin;
		fin.open(infile);
		fin>>nc0>>nr0;
		printf("nc = %d\tnr=%d\n",nc0,nr0);
	
		_int nc = ceil((_float)nc0/(_float)num_files)+1;
		_int ctr = 0;
		ofstream* fout = new ofstream[num_files]();
		for(_int i=0;i<num_files;i++)
		{
			fout[i].open(fname[i]);
			if ((nc0-ctr)>nc)
				fout[i]<<nc<<" "<<nr0<<endl;
			else
				fout[i]<<nc0-ctr<<" "<<nr0<<endl;
			ctr = ctr+nc;
		}
		
		fin.ignore();
		_int file_no = -1;
		for(_int i=0; i<nc0; i++)
		{
			if ((i%100000)==0)
				printf("%d\n",i);
			if ((i%nc) == 0)
				file_no++;
			
			string line;
			getline(fin,line);
		fout[file_no]<<line<<endl;
		}
		fin.close();
		for(_int i =0;i<num_files;i++)
			fout[i].close();
	}
}

inline void calc_knn_sparse_fast(IMat* trn_ft_mat, IMat* tst_ft_mat, SMatF* score_mat, _float threshold, int num_threads)
{
	_int num_ft = tst_ft_mat->nr;
	_int num_trn = trn_ft_mat->nc;
	_int num_tst = tst_ft_mat->nc;
	
	IMat* inv_ind = trn_ft_mat->transpose();

	printf("num_trn = %d\tnum_tst=%d\tnum_ft=%d\n",num_trn,num_tst,num_ft);	
	score_mat->nr = num_trn;
	score_mat->nc = num_tst;
	score_mat->size = new _int[num_tst]();
	score_mat->data = new pairIF*[num_tst];
	omp_set_dynamic(0);
	omp_set_num_threads(num_threads);
	#pragma omp parallel shared(trn_ft_mat,tst_ft_mat,num_tst)
	{
	#pragma omp for
	for(_int i=0; i<num_tst; i++)
	{
		if ((i%1000)==0)
			printf("%d\n",i);
		
		map<_int,_int> sim;
		_int ind = 0;
		for(_int j=0; j<tst_ft_mat->size[i]; j++)
		{
			_int ft = tst_ft_mat->data[i][j];
			for(_int k=0; k<inv_ind->size[ft]; k++)
			{
				ind = inv_ind->data[ft][k];
				if (sim.count(ind)==0)
					sim[ind] = 1;
				else
					sim[ind] = sim[ind]+1;
			}
		}
		score_mat->data[i] = new pairIF[sim.size()];
		_float prod = 0;
		_int ctr = 0;
		for (std::map<_int,_int>::iterator it=sim.begin(); it!=sim.end(); ++it)
		{
			prod = (_float)it->second/(_float)tst_ft_mat->size[i];
			if(prod>=threshold)
			{
				score_mat->data[i][ctr].first = it->first;
				score_mat->data[i][ctr].second = prod;
				ctr++;
			}
		}
		Realloc(sim.size(),ctr,score_mat->data[i]);
		score_mat->size[i] = ctr;
	}
	}
	delete inv_ind;
}

