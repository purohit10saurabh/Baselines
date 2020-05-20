INC=-I ../Tools/
CXXFLAGS=-std=c++11 -O3
LIBFLAGS=-pthread -fopenmp

all: clean parabel_train parabel_predict
compare_trees_clean : rm_compare_trees compare_trees

compare_trees:
	$(CXX) -o compare_trees $(CXXFLAGS) $(INC) compare_trees.cpp parabel.cpp ../Tools/*.cpp $(LIBFLAGS)

convert_tree_format:
	$(CXX) -o convert_tree_format $(CXXFLAGS) $(INC) convert_tree_format.cpp parabel.cpp ../Tools/*.cpp $(LIBFLAGS)

parabel_train:
	$(CXX) -o parabel_train $(CXXFLAGS) $(INC) parabel_train.cpp parabel.cpp ../Tools/*.cpp $(LIBFLAGS)

parabel_predict:
	$(CXX) -o parabel_predict $(CXXFLAGS) $(INC) parabel_predict.cpp parabel.cpp ../Tools/*.cpp $(LIBFLAGS)

clean:
	rm -f parabel_train parabel_predict

rm_compare_trees:
	rm -f compare_trees
