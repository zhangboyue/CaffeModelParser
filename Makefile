SRC_FILES = cmparser.cc

GXX_LINK_OPTS := -lpthread -lprotobuf -lpthread
cmparser : $(SRC_FILES)
	g++ -g $< caffe.pb.o -o $@ $(GXX_LINK_OPTS) 

clean: 
	rm -f cmparser
