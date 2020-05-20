
git clone https://github.com/rupea/LabelFilters.git
cd LabelFilters
cwd=`pwd`

export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python2.7/"
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar -xvzf boost_1_69_0.tar.gz 
mv boost_1_69_0 boost
cd boost/
sh bootstrap.sh
./b2
cd $cwd
# The Boost C++ Libraries were successfully built!

# The following directory should be added to compiler include paths:

#      

# The following directory should be added to linker library paths:

#     /data/home/anshumitts/scratch/lab/xc_dl/baselines/programs/LabelFilters/Boost/stage/lib


git clone https://github.com/gperftools/gperftools.git
sudo apt-get install libunwind-dev
cd gperftools
sh autogen.sh
./configure --prefix=<LOCAL DIRECTORY>
make
sudo make install
cd $cwd

git clone https://github.com/eigenteam/eigen-git-mirror.git
mv eigen-git-mirror Eigen
cd Eigen/Eigen
mkdir -p build
cd build
cmake ..
make
sudo make install
cd $cwd

git clone https://github.com/RoaringBitmap/CRoaring.git
cd CRoaring
mkdir -p build
cd build
cmake ..
make
sudo make install
cd $cwd

cd src
make CRoaring
make

sudo apt install numactl
