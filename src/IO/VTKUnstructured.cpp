/*---------------------------------------------------------------------------*\
 *
 *  bitpit
 *
 *  Copyright (C) 2015-2017 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of bitpit.
 *
 *  bitpit is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  bitpit is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with bitpit. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#include "VTK.hpp"

namespace bitpit{

/*!
 * @class VTKUnstructuredGridStreamer
 * @ingroup VisualizationToolKit
 * @brief Streamer for VTKUnstructuredGrid if the grid is made of homogeneous types
 */

/*!
 * Sets the type of and numer of elements to be written
 * @param[in] type element type
 * @param[in] n number of cells in grid
 */
void VTKUnstructuredGridStreamer::setGrid( VTKElementType type, long n){
    m_homogeneousType = type ;
    m_cells = n ;

}

/*!
 * Writes data to stream 
 * @param[in] str file stream for writing
 * @param[in] name name of field
 * @param[in] format ASCII or BINARY format
 */
void VTKUnstructuredGridStreamer::flushData( std::fstream &str, std::string name, VTKFormat format){

    assert( m_homogeneousType != VTKElementType::UNDEFINED ) ;

    if( format == VTKFormat::APPENDED){

        if(name == "types" ){
            uint8_t type = (uint8_t) m_homogeneousType ;
            for( unsigned int i=0; i<m_cells; ++i){
                genericIO::flushBINARY(str, type );
            }
        
        } else if(name == "offsets" ){
            uint8_t     n = vtk::getElementNodeCount(m_homogeneousType) ;
            uint64_t    offset(0) ;
            for( unsigned int i=0; i<m_cells; ++i){
                offset += n ;
                genericIO::flushBINARY(str, offset );
            }
        
        }

    } else {
        if(name == "types" ){
            uint8_t type = (uint8_t) m_homogeneousType ;
            for( unsigned int i=0; i<m_cells; ++i)
                genericIO::flushASCII(str, type );
        
        } else if(name == "offsets" ){
            uint8_t     n = vtk::getElementNodeCount(m_homogeneousType) ;
            uint64_t    offset(0) ;
            for( unsigned int i=0; i<m_cells; ++i){
                offset += n ;
                genericIO::flushASCII(str, offset );
            }
        
        }

    }

}

/*!
 * @class VTKUnstructuredGrid
 * @ingroup VisualizationToolKit
 * @brief VTK input output for Unstructured Meshes
 *
 * VTKUnstructuredGrid provides methods to read and write parallel and serial unstructured meshes and data. 
 * The class is agnostic with respect to the container used for the data and provides an interface through the CRTP mechanism.
 *
 */

/*!  
 *  Destructor.
 */
VTKUnstructuredGrid::~VTKUnstructuredGrid( ) {

}

/*!  
 *  Default constructor.
 *  Allocates four geometry fields called "Points"(Float64), "offsets"(Int32), "types"(Int32) and "connectivity"(Int32).
 */
VTKUnstructuredGrid::VTKUnstructuredGrid( ) :VTK() {

    m_fh.setAppendix("vtu");
    m_homogeneousType = VTKElementType::UNDEFINED ;

    m_geometry.push_back( VTKField("Points") ) ;
    m_geometry.push_back( VTKField("offsets") ) ;
    m_geometry.push_back( VTKField("types") ) ;
    m_geometry.push_back( VTKField("connectivity") ) ;

    for( auto & field : m_geometry ){
        field.setLocation( VTKLocation::CELL ) ;
        field.setFieldType( VTKFieldType::KNOWN_BY_CLASS ) ;
        field.setDataType( VTKDataType::Int32 ) ;
        field.setCodification(m_geomCodex);
    }

    m_geometry[0].setLocation( VTKLocation::POINT ) ;
    m_geometry[0].setFieldType( VTKFieldType::VECTOR ) ;
    m_geometry[0].setDataType( VTKDataType::Float64 ) ;

}

/*!  
 *  Constructor.
 *  sets input parameters and calls default constructor
 *  @param[in] dir  Directory of vtk file with final "/"
 *  @param[in] name Name of vtk file without suffix
 */
VTKUnstructuredGrid::VTKUnstructuredGrid( std::string dir, std::string name ):VTKUnstructuredGrid( ){

    setNames( dir, name ) ; 

}

/*!  
 *  Constructor for grid with homogeneous element type
 *  sets input parameters and calls default constructor
 *  @param[in] dir Directory of vtk file with final "/"
 *  @param[in] name Name of vtk file without suffix
 *  @param[in] type Type of element
 */
VTKUnstructuredGrid::VTKUnstructuredGrid( std::string dir, std::string name, VTKElementType type ):VTKUnstructuredGrid( dir, name ){

    setElementType( type ) ;

}

/*!  
 *  Tell VTKUnstructuredGrid that grid is made homogeously of one element type; 
 *  Consequently type and offset information are handled directly in class and need not to be provided via interface
 *  @param[in] type Type of element in grid
 */
void VTKUnstructuredGrid::setElementType( VTKElementType type ){

    m_homogeneousType = type ;

    m_geometry[1].setDataType( VTKDataType::UInt64) ; 
    m_geometry[1].setStreamer(m_unstructuredStreamer) ;

    m_geometry[2].setDataType( VTKDataType::UInt8) ; 
    m_geometry[2].setStreamer(m_unstructuredStreamer) ;


}

/*!  
 *  sets the size of the unstructured grid. 
 *  If VTKUnstructuredGrid::setElementType(VTKElelementType) has been called the last argument can be omitted and the connectivity size will be calculated within the method.
 *  @param[in] ncells number of cells
 *  @param[in] npoints number of points
 *  @param[in] nconn size of the connectivity information;
 */
void VTKUnstructuredGrid::setDimensions( uint64_t ncells, uint64_t npoints, uint64_t nconn ){

    m_cells        = ncells ;
    m_points       = npoints ;
    m_connectivity  = nconn ;

    if( m_homogeneousType != VTKElementType::UNDEFINED ){
        m_connectivity = ncells *vtk::getElementNodeCount( m_homogeneousType ) ;
        m_unstructuredStreamer.setGrid(m_homogeneousType, ncells );
    }

}

/*!  
 *  sets the size of the unstructured grid for a homogenous grid.
 *  @param[in] ncells number of cells
 *  @param[in] npoints number of points
 *  @param[in] type typeof element in grid
 */
void VTKUnstructuredGrid::setDimensions( uint64_t ncells, uint64_t npoints, VTKElementType type ){

    setElementType( type );
    setDimensions( ncells, npoints );

}

/*!
 * Associates streamer to a geometrical field
 * @param[in] fieldEnum which geometrical field 
 * @param[in] streamer VTKBaseStreamer
 */
void VTKUnstructuredGrid::setGeomData( VTKUnstructuredField fieldEnum, VTKBaseStreamer *streamer ){

    int      index = static_cast<int>(fieldEnum) ;
    VTKField& field = m_geometry[index] ;

    field.setStreamer( *streamer ) ;

}

/*!
 * Associates streamer to a geometrical field
 * @param[in] fieldEnum which geometrical field 
 * @param[in] type type of data [ VTKDataType::[[U]Int[8/16/32/64] / Float[32/64] ] ]
 * @param[in] streamer pointer to data streamer
 */
void VTKUnstructuredGrid::setGeomData( VTKUnstructuredField fieldEnum, VTKDataType type, VTKBaseStreamer *streamer ){

    int      index = static_cast<int>(fieldEnum) ;
    VTKField& field = m_geometry[index] ;

    field.setDataType( type ) ;
    field.setStreamer( *streamer ) ;


}

/*!  
 *  Reads "type" information of existing grid and calculates the correspondng connectivity size.
 *  @return size of the connectivity information
 */
uint64_t VTKUnstructuredGrid::calcSizeConnectivity( ){

    uint64_t                 nconn(0) ;

    std::fstream             str  ;
    std::fstream::pos_type   position_appended;
    std::string              line;
    char                     c_ ;
    uint32_t                 nbytes32 ;
    uint64_t                 nbytes64 ;

    str.open( m_fh.getPath( ), std::ios::in ) ;

    //Read appended data
    //Go to the initial position of the appended section
    while( getline(str, line) && (! bitpit::utils::keywordInString( line, "<AppendedData")) ){}

    str >> c_;
    while( c_ != '_') str >> c_;

    position_appended = str.tellg();


    str.close();
    str.clear();

    //Open in binary for read
    str.open( m_fh.getPath( ), std::ios::in | std::ios::binary);

    if( m_geometry[3].getCodification() == VTKFormat::APPENDED ){
        str.seekg( position_appended) ;
        str.seekg( m_geometry[3].getOffset(), std::ios::cur) ;

        if( m_headerType== "UInt32") {
            genericIO::absorbBINARY( str, nbytes32 ) ;
            nconn = nbytes32 /VTKTypes::sizeOfType( m_geometry[3].getDataType() ) ;
        }

        if( m_headerType== "UInt64") {
            genericIO::absorbBINARY( str, nbytes64 ) ;
            nconn = nbytes64 /VTKTypes::sizeOfType( m_geometry[3].getDataType() ) ;
        }
    }


    //Read geometry
    if(  m_geometry[3].getCodification() == VTKFormat::ASCII ){
        str.seekg( m_geometry[3].getPosition() ) ;

        std::string              line ;
        std::vector<uint64_t>    temp;

        nconn = 0 ;

        getline( str, line) ;
        while( ! bitpit::utils::keywordInString(line,"/DataArray") ) {

            temp.clear() ;
            bitpit::utils::convertString( line, temp) ;
            nconn += temp.size() ;
        }


    }

    str.close();

    return nconn ;


}

/*!  
 *  Writes entire VTU but the data.
 */
void VTKUnstructuredGrid::writeMetaInformation( ){

    std::fstream str ;
    std::string line ; 

    str.open( m_fh.getPath( ), std::ios::out ) ;

    //Writing XML header
    str << "<?xml version=\"1.0\"?>" << std::endl;

    //Writing Piece Information
    str << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"" << m_headerType << "\">" << std::endl;
    str << "  <UnstructuredGrid>"  << std::endl;;
    str << "    <Piece  NumberOfPoints=\"" << m_points << "\" NumberOfCells=\"" << m_cells << "\">" << std::endl;

    //Header for Data
    writeDataHeader( str, false );

    //Wring Geometry Information
    str << "      <Points>" << std::endl ;;
    writeDataArray( str, m_geometry[0] ) ;
    str << "      </Points>" << std::endl;

    str << "      <Cells>" << std::endl ;;
    writeDataArray( str, m_geometry[1] ) ;
    writeDataArray( str, m_geometry[2] ) ;
    writeDataArray( str, m_geometry[3] ) ;
    str << "      </Cells>" << std::endl;

    //Closing Piece
    str << "    </Piece>" << std::endl;
    str << "  </UnstructuredGrid>"  << std::endl;

    //Appended Section

    str << "  <AppendedData encoding=\"raw\">" << std::endl;
    str << "_" ;
    str << std::endl ;
    str << "</VTKFile>" << std::endl;

    str.close() ;

}

/*!  
 *  Writes collection file for parallel output. 
 *  Is called by rank 0 in VTK::Write()
 */
void VTKUnstructuredGrid::writeCollection( ){

    std::fstream str ;

    FileHandler     fhp, fho ;

    fhp = m_fh ;
    fho = m_fh ;

    fhp.setParallel(false) ;
    fhp.setAppendix("pvtu") ;

    fho.setDirectory(".") ;

    str.open( fhp.getPath( ), std::ios::out ) ;

    //Writing XML header
    str << "<?xml version=\"1.0\"?>" << std::endl;

    //Writing Piece Information
    str << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    str << "  <PUnstructuredGrid GhostLevel=\"0\">"  << std::endl;;

    //Header for Data
    writeDataHeader( str, true );

    //Wring Geometry Information
    str << "      <PPoints>" << std::endl;
    writePDataArray( str, m_geometry[0] ) ;
    str << std::endl ;
    str << "      </PPoints>" << std::endl;


    for( int i=0; i<m_procs; i++){
        fho.setBlock(i) ;
        str << "    <Piece  Source=\"" << fho.getPath() <<  "\"/>" << std::endl;
    }

    str << "  </PUnstructuredGrid>"  << std::endl;
    str << "</VTKFile>" << std::endl;

    str.close() ;


}

/*!  
 *  Reads meta data of VTU file (grid size, data fields, codex, position of data within file).
 *  Calls setDimension.
 */
void VTKUnstructuredGrid::readMetaInformation( ){

    std::fstream str;
    std::string line, temp;

    std::fstream::pos_type        position;

    str.open( m_fh.getPath( ), std::ios::in ) ;

    getline( str, line);
    while( ! bitpit::utils::keywordInString( line, "<VTKFile")){
        getline(str, line);
    }

    if( bitpit::utils::getAfterKeyword( line, "header_type", '\"', temp) ){
        setHeaderType( temp) ;
    }

    while( ! bitpit::utils::keywordInString( line, "<Piece")){
        getline(str, line);
    }

    bitpit::utils::getAfterKeyword( line, "NumberOfPoints", '\"', temp) ;
    bitpit::utils::convertString( temp, m_points );

    bitpit::utils::getAfterKeyword( line, "NumberOfCells", '\"', temp) ;
    bitpit::utils::convertString( temp, m_cells );


    position = str.tellg() ;
    readDataHeader( str ) ;


    for( auto &field : m_geometry ){ 
        str.seekg( position) ;
        if( ! readDataArray( str, field ) ) {
            log::cout() << field.getName() << " DataArray not found" << std::endl ;
        }
    }


    str.close() ;

    if( m_homogeneousType == VTKElementType::UNDEFINED) {
        setDimensions( m_cells, m_points, calcSizeConnectivity() ) ;
    } else {
        // Metadata information read form file may not match the information
        // set in our own streamer. If the grid is homogeneous, we need to
        // reset all metadata that can't be overwritten.
        setElementType(m_homogeneousType) ;

        // Set the dimension of the grid
        setDimensions( m_cells, m_points ) ;
    }


}

/*!
 * Calculates the size (in bytes) of a field
 * @param[in] field field 
 * @return size of the field
 */
uint64_t VTKUnstructuredGrid::calcFieldSize( const VTKField &field ){

    uint64_t bytes = calcFieldEntries(field) ;
    bytes *= VTKTypes::sizeOfType( field.getDataType() ) ;

    return bytes ;

}

/*!
 * Calculates the number of entries of a field
 * @param[in] field field 
 * @return size of the field
 */
uint64_t VTKUnstructuredGrid::calcFieldEntries( const VTKField &field ){

    uint64_t entries(0) ;
    std::string name( field.getName() ) ;

    if( name == "Points" ){
        entries = m_points *static_cast<int>(VTKFieldType::VECTOR) ; 

    } else if( name == "offsets" ){
        entries = m_cells ;

    } else if( name == "types" ){
        entries = m_cells ;

    } else if( name == "connectivity"){
        entries = m_connectivity ;

    } else{

        VTKLocation location( field.getLocation() ) ;
        assert( location != VTKLocation::UNDEFINED) ;

        if( location == VTKLocation::CELL ){
            entries = m_cells ;

        } else if( location == VTKLocation::POINT ){
            entries = m_points ;

        }

        VTKFieldType fieldType( field.getFieldType() ) ;
        assert( fieldType != VTKFieldType::UNDEFINED) ;

        entries *= static_cast<uint64_t>(fieldType) ;

    }

    return entries ;

}

/*!
 * Calculates the compnents of a field
 * @param[in] field field 
 * @return size of the field
 */
uint8_t VTKUnstructuredGrid::calcFieldComponents( const VTKField &field ){

    uint8_t comp ;
    std::string name( field.getName() ) ;

    if( name == "Points" ){
        comp = static_cast<int>(VTKFieldType::VECTOR) ; 

    } else if( name == "offsets" ){
        comp = 1 ;

    } else if( name == "types" ){
        comp = 1 ;

    } else if( name == "connectivity" ){
       if( m_homogeneousType != VTKElementType::UNDEFINED){
            comp = vtk::getElementNodeCount( m_homogeneousType ) ;

       } else {
           comp = 1;

       }

    } else{

        VTKFieldType fieldType( field.getFieldType() ) ;
        assert( fieldType != VTKFieldType::UNDEFINED) ;

        comp = static_cast<uint8_t>(fieldType) ;

    }

    return comp ;

}

/*!  
 *  Returns the size of the connectivity information
 *  @return size of connectivity
 */
uint64_t VTKUnstructuredGrid::getNConnectivity( ){

    return m_connectivity ;
}

}
