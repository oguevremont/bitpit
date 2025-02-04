/*---------------------------------------------------------------------------*\
 *
 *  bitpit
 *
 *  Copyright (C) 2015-2021 OPTIMAD engineering Srl
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

#ifndef __BITPIT_BINARY_STREAM_HPP__
#define __BITPIT_BINARY_STREAM_HPP__

#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace bitpit {

// Stream operators
class IBinaryStream;
class OBinaryStream;

template<typename T>
IBinaryStream & operator>>(IBinaryStream &stream, T &value);

template<typename T>
IBinaryStream& operator>>(IBinaryStream &stream, std::vector<T> &vector);

template<>
IBinaryStream & operator>>(IBinaryStream &stream, std::string &value);

template<typename T>
OBinaryStream & operator<<(OBinaryStream &stream, const T &value);

template<typename T>
OBinaryStream& operator<<(OBinaryStream &stream, const std::vector<T> &vector);

template<>
OBinaryStream & operator<<(OBinaryStream &stream, const std::string &value);

// Binary stream
class BinaryStream {

public:
    virtual ~BinaryStream() = default;

    bool eof() const;

    std::streampos tellg() const;
    bool seekg(std::size_t pos);
    bool seekg(std::streamoff offset, std::ios_base::seekdir way);

    char * data();
    const char * data() const;

    virtual void setSize(std::size_t size);
    std::size_t getSize() const;

    std::size_t getCapacity() const;

    int getChunkSize();
    int getChunkCount();

protected:
    std::vector<char> m_buffer;
    std::size_t m_size;
    std::size_t m_pos;

    BinaryStream();
    BinaryStream(std::size_t capacity);
    BinaryStream(const char *buffer, std::size_t capacity);
    BinaryStream(const std::vector<char> &buffer);

    void open(const char *buffer, std::size_t capacity);
    void open(std::size_t capacity);

    void setCapacity(std::size_t capacity);

private:
    int m_chunkSize;

};

class IBinaryStream : public BinaryStream {

template<typename T>
friend IBinaryStream & (operator>>)(IBinaryStream &stream, T &value);

public:
    IBinaryStream(void);
    IBinaryStream(std::size_t size);
    IBinaryStream(const char *buffer, std::size_t size);
    IBinaryStream(const std::vector<char> &buffer);

    void open(const char *buffer, std::size_t size);
    void open(std::size_t size);

    void read(char *data, std::size_t size);

};

class OBinaryStream : public BinaryStream {

template<typename T>
friend OBinaryStream & (operator<<)(OBinaryStream &stream, const T &value);

public:
    OBinaryStream();
    OBinaryStream(std::size_t size);

    void open(std::size_t size);

    void setSize(std::size_t size) override;
    void squeeze();

    void write(const char *data, std::size_t size);

private:
    bool m_expandable;

};

}

// Template implementation
#include "binary_stream.tpp"

#endif
