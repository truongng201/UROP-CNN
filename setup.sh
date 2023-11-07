DATA_DIR='data'

if [ ! -d $DATA_DIR ]; then
  mkdir $DATA_DIR
  cd /content/data
  wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm -f cifar-10-python.tar.gz
else
  echo "Directory $DATA_DIR already exists."
fi

