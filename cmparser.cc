#include <fstream>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <assert.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "caffe.pb.h"

//---------------------------------------------------------------------------------------------------------------------
typedef struct weight_desc_t {
    const char *host_layer;
    int host_layer_idx;
    int blob_idx;
} weight_desc;

const char *outdir = "./caffedata";
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;


//---------------------------------------------------------------------------------------------------------------------
static bool readProtobufFromBinaryFile( const char *filename, Message *proto, FILE *flog );
static bool parseNetParameter( caffe::NetParameter *protoNetParam, FILE *flog );
static bool parseLayerParamter( const caffe::LayerParameter *protoLayerParam, int layer_idx, FILE *flog );
static bool parseBlobProto( const caffe::BlobProto *proto_blob, const weight_desc *wdesc, FILE *flog );


//---------------------------------------------------------------------------------------------------------------------
#define LOG( file, ... )  \
    do { \
        if ( file ) fprintf( file, __VA_ARGS__ ); \
        printf( __VA_ARGS__ ); \
    } while( 0 )


//---------------------------------------------------------------------------------------------------------------------
void printHelp( void )
{
const char *helpMesg = "\n\
Desc:\n\
    cmparser is used for parsing binary caffemodel, output will be placed at `caffedata' in the current working directory.\n\
\n\
Syntax:\n\
    To parse caffemodule( %> stands for shell prompt, and angle braket '<' '>' are not necessary for typing in ):\n\
        %> cmparser -f <filename>\n\
        e.g. %> cmparser -f ./alextnet.caffemodel\n\
\n\
    To generate help message\n\
        %> cmparser -h\n\
";
    printf( "%s", helpMesg );
}

int main( int argc, const char * argv[] )
{
    int i, format;
    FILE *flog;
    caffe::NetParameter rootProto;
    const char *fname = NULL;
    char buffer[BUFSIZ];

    // 1. Create the output directory
    snprintf( buffer, BUFSIZ, "mkdir %s\n", outdir );
    system( buffer );

    // 2. Parse input arguments
    if ( argc < 2 ) {
        printf( "[Error]: Invalid syntax, not enought arguments!\n" );
        printHelp( );
        return 1;
    }

    for ( i = 1; i < argc; i++ ) {
        if ( !strcmp( argv[i], "-h" ) ) {
            printHelp( );
            return 0;
        }

        if ( !strcmp( argv[i], "--file-name" ) || !strcmp( argv[i], "-f" ) )
            fname = argv[++i];
        else {
           printf( "Warning: Unsupported option: %s found, the rest options are ignored!!!\n",
                 argv[i] );
           break;
        }
    }

    // 3. Check necessary arguments
    if ( !fname ) {
        printf( "[Error]: -f option is not specified. To view the compte help message type in:\n " );
        printf( "    cmparser -h\n" );
        return 1;
    }


    // 4. Do parsing protot messages
    snprintf( buffer, BUFSIZ, "%s/cmparser.log", outdir );
    flog = fopen( buffer, "w" );
    if ( !flog ) {
        printf( "[Error]: Failed to open log `%s' for write!!!\n", buffer );
        return 1;
    }

    if ( !readProtobufFromBinaryFile( fname, &rootProto, flog ) ) {
        fclose( flog );
        return 2;
    }

    parseNetParameter( &rootProto, flog );
    fclose( flog );
    return 0;
}


//---------------------------------------------------------------------------------------------------------------------
bool 
parseLayerParamter( const caffe::LayerParameter *protoLayerParam, int layer_idx, FILE *flog )
{
    int i, n_bottom, n_top, n_blob, width=20;
    const caffe::BlobProto *proto_blob;
    weight_desc wdesc = { NULL, layer_idx, -1 };
    assert( protoLayerParam );
    n_bottom = protoLayerParam->bottom_size( );
    n_top = protoLayerParam->top_size( );
    n_blob = protoLayerParam->blobs_size( );
    LOG( flog, "\n[State]: Parsing Layer-%d\n", layer_idx );
    LOG( flog, "  %-*s=  %s\n", width, "name", ( protoLayerParam->has_name( ) ? protoLayerParam->name( ).c_str( ) : "N/A" ) );
    LOG( flog, "  %-*s=  %s\n", width, "type", ( protoLayerParam->has_type( ) ? protoLayerParam->type( ).c_str( ) : "N/A" ) );
    LOG( flog, "  %-*s=  %d\n", width, "bottom_size", n_bottom );
    LOG( flog, "  %-*s=  %d\n", width, "top_size", n_top );
    LOG( flog, "  %-*s=  %d\n", width, "blob_size", n_blob );

    for ( i = 0; i < n_blob; i++ )  {
        LOG( flog, "  ---- Blob-%d ----\n", i );
        proto_blob = &protoLayerParam->blobs( i );
        wdesc.blob_idx  = i;
        wdesc.host_layer = ( protoLayerParam->has_name( ) ? protoLayerParam->name( ).c_str( ) : NULL );
        parseBlobProto( proto_blob,  &wdesc, flog );
    }
}

bool
parseNetParameter( caffe::NetParameter *protoNetParam, FILE *flog )
{
    int i, width = 20; 
    const ::caffe::LayerParameter *protoLayerParam;
    assert( protoNetParam );
    LOG( flog, "[State]: Parsing net parameters...\n" );

    if ( protoNetParam->has_name( ) ) {
        const char *netName = protoNetParam->name( ).c_str( );
        LOG( flog, "  %-*s=  %s\n", width, "name", netName );
    } else
        LOG( flog, "  %-*s=  N\\A\n", width, "name" );
    LOG( flog, "  %-*s=  %d\n", width, "input_size", protoNetParam->input_size( ) );
    LOG( flog, "  %-*s=  %d\n", width, "input_shape_size",  protoNetParam->input_shape_size( ) );
    LOG( flog, "  %-*s=  %d\n", width, "input_dim_size", protoNetParam->input_dim_size( ) );
    LOG( flog, "  %-*s=  %d\n", width, "has_state", protoNetParam->has_state( ) );
    LOG( flog, "  %-*s=  %d\n", width, "has_debug_info", protoNetParam->has_debug_info( ) );
    LOG( flog, "  %-*s=  %d\n", width, "has_forced_backword", protoNetParam->has_force_backward( ) );
    LOG( flog, "  %-*s=  %d\n", width, "layer_size", protoNetParam->layer_size( ) );
    LOG( flog, "  %-*s=  %d\n", width, "layers_size", protoNetParam->layers_size( ) );

    LOG( flog, "[State]: Parsing layer parameters...\n" );
    for ( i = 0; i < protoNetParam->layer_size( ); i++ ) {
        protoLayerParam = &protoNetParam->layer( i ); 
        parseLayerParamter( protoLayerParam, i, flog );
    }

    return true;
}

bool
readProtobuFromTextFile( const char *filename, Message *proto )
{
    int fd = open( filename, O_RDONLY );
    if ( fd < 0 ) {
        printf( "Error: Failed to open '%s'\n", filename );
        return false;
    }

    FileInputStream *input = new FileInputStream( fd );
    bool success = google::protobuf::TextFormat::Parse( input, proto );

    delete input;
    close( fd );

    return success;
}

bool
readProtobufFromBinaryFile( const char *filename, Message *proto, FILE *flog )
{
    LOG( flog, "[Info]: Reading proto message from '%s'...\n", filename );
    int fd = open( filename, O_RDONLY );
    if ( fd < 0 ) {
        LOG( flog, "[Error]: Failed to open file `%s'!!!\n", filename );
        return false;
    }

    ZeroCopyInputStream *raw_input = new FileInputStream( fd );
    CodedInputStream *coded_input = new CodedInputStream( raw_input );
    coded_input->SetTotalBytesLimit( INT_MAX, 536870912 ); 

    bool success = proto->ParseFromCodedStream( coded_input );

    delete coded_input;
    delete raw_input;
    close( fd );

    return success;
}

bool
parseBlobProto( const caffe::BlobProto *proto_blob, const weight_desc *wdesc, FILE *flog )
{
    FILE *fdump;
    float data;
    char buffer[BUFSIZ], layername[512];
    int i, n_dim, offset = 0, width = 20, data_size;
    const caffe::BlobShape *blob_shape;
    assert( proto_blob );

    if ( proto_blob->has_shape( ) ) {
        blob_shape = &proto_blob->shape( );
        n_dim = blob_shape->dim_size( );
        offset += snprintf( buffer + offset, BUFSIZ - offset, "[" );
        for ( i = 0; i < n_dim; i++ )
            offset += snprintf( buffer + offset, BUFSIZ - offset, "%lu, ", blob_shape->dim( i ) );
        offset += snprintf( buffer + offset, BUFSIZ - offset, "\b\b]" );
        LOG( flog, "    %-*s= %s\n", width, "shape", buffer );
    } else
        LOG( flog, "    %-*s= %s\n", width, "shape", "N/A" );
    if ( proto_blob->has_num( ) )
        LOG( flog, "    %-*s= %d\n", width, "num", proto_blob->num( ) );
    else
        LOG( flog, "    %-*s= %s\n", width, "num", "N/A" );

    if ( proto_blob->has_channels( ) )
        LOG( flog, "    %-*s= %d\n", width, "channel", proto_blob->channels( ) );
    else
        LOG( flog, "    %-*s= %s\n", width, "channel", "N/A" );

    if ( proto_blob->has_height( ) )
        LOG( flog, "    %-*s= %d\n", width, "height", proto_blob->height( ) );
    else
        LOG( flog, "    %-*s= %s\n", width, "height", "N/A" );

    if ( proto_blob->has_width( ) )
        LOG( flog, "    %-*s= %d\n", width, "width", proto_blob->width( ) );
    else
        LOG( flog, "    %-*s= %s\n", width, "width", "N/A" );
    LOG( flog, "    %-*s= %d\n", width, "data_size", proto_blob->data_size( ) );
    LOG( flog, "    %-*s= %d\n", width, "double_data_size", proto_blob->double_data_size( ) );

    if ( wdesc ) { // We're dumping a weight-blob for a layer now
        // 1. Write weight.txt
        if ( wdesc->host_layer ) {
            strncpy( layername, wdesc->host_layer, 512 ); 
            for ( i = 0; i < strlen( layername ); i++ ) {
                if ( layername[i] != '/' )
                    continue;
                layername[i] = '#';
            }

            snprintf( buffer, BUFSIZ, "%s/layer_%s.weight%d.txt",
                      outdir, layername, wdesc->blob_idx );
        } else
            snprintf( buffer, BUFSIZ, "%s/layer_%d.weight%d.txt",
                      outdir, wdesc->host_layer_idx, wdesc->blob_idx );
        data_size = proto_blob->data_size( );
        fdump = fopen( buffer, "w" ); 
        if ( !fdump )
            LOG( flog, "[Error]: Failed to open `%s' for write!!!\n", buffer );

        LOG( flog, "[State]: Writing weight to `%s'...\n", buffer );
        for ( i = 0; i < data_size; i++ ) 
            fprintf( fdump, "%.8f\n", proto_blob->data( i ) );
        fclose( fdump );

        // 2. Write weight.data
        if ( wdesc->host_layer ) {
            snprintf( buffer, BUFSIZ, "%s/layer_%s.weight%d.data",
                      outdir, layername, wdesc->blob_idx );
        } else
            snprintf( buffer, BUFSIZ, "%s/layer_%d.weight%d.data",
                      outdir, wdesc->host_layer_idx, wdesc->blob_idx );

        fdump = fopen( buffer, "wb" ); 
        LOG( flog, "[State]: Writing weight to `%s'...\n", buffer );
        for ( i = 0; i < data_size; i++ ) {
            data = proto_blob->data( i );
            fwrite( (void *) &data, sizeof( float ), 1, fdump ); 
            fprintf( fdump, "%.8f\n", proto_blob->data( i ) );
        }
        fclose( fdump );
    }
    return true;
}
