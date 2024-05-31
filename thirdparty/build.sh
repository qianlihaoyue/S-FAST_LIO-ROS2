unzip Sophus.zip
cd Sophus

if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../../sophus
make install

cd ../../
rm -rf Sophus