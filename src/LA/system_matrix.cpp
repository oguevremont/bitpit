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

#include "system_matrix.hpp"

namespace bitpit {

/**
* \class SparseMatrix
* \ingroup system_solver
*
* \brief Sparse matrix.
*
* Rather than working with individual elements in the matrix, it is possible
* to employ blocks of elements. The size of the blocks can be defined during
* assembly. When a size different that one is provided, the matrix will store
* elements by fixed-sized dense nb × nb blocks, where nb is the size of the
* blocks. Blocking may be advantageous when solving PDE-based simulations
* that leads to matrices with a naturally blocked structure (with a block size
* equal to the number of degrees of freedom per cell).
*
* When blocking is used, row and column indexes will count the number of blocks
* in the row/column direction, not the number of rows/columns of the matrix.
*/
#if BITPIT_ENABLE_MPI==1
/**
*
* The SparseMatrix class mimics the beahviour of the sparse parallel matrix
* in AIJ/BAIJ format of the PETSc library.
*
* The parallel matrix is partitioned across processes such that the first
* m0 rows belong to process 0, the next m1 rows belong to process 1, the
* next m2 rows belong to process 2 etc.. where m0, m1, m2,.. are the local
* number of rows defined by each processes.
*
* Each process stores values corresponding to [m x N] submatrix, wher N
* is the global number of columns of the matrix, N is obtained summing the
* local number of columns of each process.
*
* Although each process stores values for the whole global row, columns are
* logically partitioned across processes, with the n0 columns belonging to
* 0th partition, the next n1 columns belonging to the next partition etc..
* where n0, n1, n2... are the local number of columns defined by each
* process.
*
* The DIAGONAL portion of the local submatrix on any given process is the
* submatrix corresponding to the rows and columns m,n corresponding to the
* given process. i.e diagonal matrix on process 0 is [m0 x n0], diagonal
* matrix on process 1 is [m1 x n1] etc. The remaining portion of the local
* submatrix [m x (N-n)] constitute the OFF-DIAGONAL portion.
*
* For a square global matrix we define each process's diagonal portion to
* be its local rows and the corresponding columns (a square submatrix); each
* process's off-diagonal portion encompasses the remainder of the local
* matrix (a rectangular submatrix).
*/
#endif

#if BITPIT_ENABLE_MPI==1
/**
* Default constructor
*/
SparseMatrix::SparseMatrix()
    : SparseMatrix(MPI_COMM_SELF)
{
}

/**
* Constructor
*
* \param communicator is the MPI communicator
*/
SparseMatrix::SparseMatrix(MPI_Comm communicator)
    : m_blockSize(0), m_nRows(0), m_nCols(0), m_nNZ(0), m_maxRowNZ(0), m_lastRow(-1),
      m_assembled(false),
      m_partitioned(false),
      m_global_nRows(0), m_global_nCols(0), m_global_nNZ(0),
      m_global_maxRowNZ(0), m_global_rowOffset(0), m_global_colOffset(0)
{
    // Set the communicator
    setCommunicator(communicator);
}
#else
/**
* Default constructor
*/
SparseMatrix::SparseMatrix()
    : m_blockSize(0), m_nRows(0), m_nCols(0), m_nNZ(0), m_maxRowNZ(0), m_lastRow(-1),
      m_assembled(false)
{
}
#endif

#if BITPIT_ENABLE_MPI==1
/**
* Constructor
*
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
SparseMatrix::SparseMatrix(long nRows, long nCols, long nNZ)
    : SparseMatrix(MPI_COMM_SELF, false, 1, nRows, nCols, nNZ)
{
}

/**
* Constructor
*
* \param blockSize is the block size of the matrix
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
SparseMatrix::SparseMatrix(int blockSize, long nRows, long nCols, long nNZ)
    : SparseMatrix(MPI_COMM_SELF, false, blockSize, nRows, nCols, nNZ)
{
}

/**
* Constructor
*
* \param communicator is the MPI communicator
* \param partitioned controls if the matrix is partitioned
* \param nRows is the number rows of the matrix, if the matrix is partitioned
* this is the number of local rows
* \param nCols is the number columns of the matrix, if the matrix is partitioned
* this is the number of local columns
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
SparseMatrix::SparseMatrix(MPI_Comm communicator, bool partitioned, long nRows, long nCols, long nNZ)
    : SparseMatrix(communicator)
{
    _initialize(partitioned, 1, nRows, nCols, nNZ);
}

/**
* Constructor
*
* \param communicator is the MPI communicator
* \param partitioned controls if the matrix is partitioned
* \param blockSize is the block size of the matrix
* \param nRows is the number rows of the matrix, if the matrix is partitioned
* this is the number of local rows
* \param nCols is the number columns of the matrix, if the matrix is partitioned
* this is the number of local columns
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
SparseMatrix::SparseMatrix(MPI_Comm communicator, bool partitioned, int blockSize, long nRows, long nCols, long nNZ)
    : SparseMatrix(communicator)
{
    _initialize(partitioned, blockSize, nRows, nCols, nNZ);
}
#else
/**
* Constructor
*
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
SparseMatrix::SparseMatrix(long nRows, long nCols, long nNZ)
    : SparseMatrix(1, nRows, nCols, nNZ)
{
}

/**
* Constructor
*
* \param blockSize is the block size of the matrix
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
SparseMatrix::SparseMatrix(int blockSize, long nRows, long nCols, long nNZ)
    : SparseMatrix()
{
    _initialize(blockSize, nRows, nCols, nNZ);
}
#endif

/*!
 * Destructor
 */
SparseMatrix::~SparseMatrix()
{
#if BITPIT_ENABLE_MPI==1
    // Free the MPI communicator
    freeCommunicator();
#endif
}

#if BITPIT_ENABLE_MPI==1
/**
* Initialize the pattern.
*
* \param partitioned controls if the matrix is partitioned
* \param nRows is the number rows of the matrix, if the matrix is partitioned
* this is the number of local rows
* \param nCols is the number columns of the matrix, if the matrix is partitioned
* this is the number of local columns
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
void SparseMatrix::initialize(bool partitioned, long nRows, long nCols, long nNZ)
{
    _initialize(partitioned, 1, nRows, nCols, nNZ);
}

/**
* Initialize the pattern.
*
* \param partitioned controls if the matrix is partitioned
* \param blockSize is the block size of the matrix
* \param nRows is the number rows of the matrix, if the matrix is partitioned
* this is the number of local rows
* \param nCols is the number columns of the matrix, if the matrix is partitioned
* this is the number of local columns
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
void SparseMatrix::initialize(bool partitioned, int blockSize, long nRows, long nCols, long nNZ)
{
    _initialize(partitioned, blockSize, nRows, nCols, nNZ);
}
#endif

/**
* Initialize the pattern.
*
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
void SparseMatrix::initialize(long nRows, long nCols, long nNZ)
{
    initialize(1, nRows, nCols, nNZ);
}

/**
* Initialize the pattern.
*
* \param blockSize is the block size of the matrix
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
void SparseMatrix::initialize(int blockSize, long nRows, long nCols, long nNZ)
{
#if BITPIT_ENABLE_MPI==1
    _initialize(false, blockSize, nRows, nCols, nNZ);
#else
    _initialize(blockSize, nRows, nCols, nNZ);
#endif
}

#if BITPIT_ENABLE_MPI==1
/**
* Internal function to initialize the pattern.
*
* \param partitioned controls if the matrix is partitioned
* \param blockSize is the block size of the matrix
* \param nRows is the number rows of the matrix, if the matrix is partitioned
* this is the number of local rows
* \param nCols is the number columns of the matrix, if the matrix is partitioned
* this is the number of local columns
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
void SparseMatrix::_initialize(bool partitioned, int blockSize, long nRows, long nCols, long nNZ)
{
    // Serial initialization
    _initialize(blockSize, nRows, nCols, nNZ);

    // Parallel initialization
    m_partitioned = partitioned;

    m_global_nRows = m_nRows;
    m_global_nCols = m_nCols;
    if (m_partitioned) {
        MPI_Allreduce(MPI_IN_PLACE, &m_global_nRows, 1, MPI_LONG, MPI_SUM, m_communicator);
        MPI_Allreduce(MPI_IN_PLACE, &m_global_nCols, 1, MPI_LONG, MPI_SUM, m_communicator);
    }

    m_global_rowOffset = 0;
    m_global_colOffset = 0;
    if (m_partitioned) {
        int nProcessors;
        MPI_Comm_size(m_communicator, &nProcessors);

        std::vector<long> nGlobalRows(nProcessors);
        MPI_Allgather(&nRows, 1, MPI_LONG, nGlobalRows.data(), 1, MPI_LONG, m_communicator);

        std::vector<long> nGlobalCols(nProcessors);
        MPI_Allgather(&nCols, 1, MPI_LONG, nGlobalCols.data(), 1, MPI_LONG, m_communicator);

        int rank;
        MPI_Comm_rank(m_communicator, &rank);
        for (int i = 0; i < rank; ++i) {
            m_global_rowOffset += nGlobalRows[i];
            m_global_colOffset += nGlobalCols[i];
        }
    }
}
#endif

/**
* Internal function to initialize the pattern.
*
* \param nRows is the number of rows of the matrix
* \param nCols is the number of columns of the matrix
* \param nNZ is the number of non-zero elements that the matrix will contain.
* If this number is unknown, an estimante (or zero) could be passed. If the
* actual number of non-zero elements turns out to be greater than the provided
* value, the initialization of the matrix will be slower because reallocation
* of internal data may be needed
*/
void SparseMatrix::_initialize(int blockSize, long nRows, long nCols, long nNZ)
{
    assert(blockSize >= 0);
    assert(nRows >= 0);
    assert(nCols >= 0);
    assert(nNZ >= 0);

    clear();

    m_blockSize = blockSize;
    m_nRows     = nRows;
    m_nCols     = nCols;

    m_pattern.reserve(m_nRows, nNZ);
    m_values.reserve(m_blockSize * m_blockSize * nNZ);
}

/**
* Clear the pattern.
*
* \param release if set to true the memory hold by the pattern will be released
*/
void SparseMatrix::clear(bool release)
{
    if (release) {
        m_pattern.clear();
        m_values.clear();
    } else {
        FlatVector2D<long>().swap(m_pattern);
        std::vector<double>().swap(m_values);
    }

    m_blockSize =  0;
    m_nRows     =  0;
    m_nCols     =  0;
    m_nNZ       =  0;
    m_maxRowNZ  =  0;
    m_lastRow   = -1;

#if BITPIT_ENABLE_MPI==1
    m_partitioned = false;

    m_global_nRows     = 0;
    m_global_nCols     = 0;
    m_global_nNZ       = 0;
    m_global_maxRowNZ  = 0;
    m_global_rowOffset = 0;
    m_global_colOffset = 0;
#endif

    m_assembled = false;
}

/*!
* Squeeze.
*
* Requests the matrix pattern to reduce its capacity to fit its size.
*/
void SparseMatrix::squeeze()
{
    m_pattern.shrinkToFit();
    m_values.shrink_to_fit();
}

/*!
* Assembly the matrix.
*
* This function should be called after adding all the rows of the matrix.
* Its purpose is to prepare the matrix for the usage.
*/
#if BITPIT_ENABLE_MPI==1
/*!
* It's a collective operation, hence it has to be called by all processes.
*/
#endif
void SparseMatrix::assembly()
{
    // Early return if the matrix is already assembled.
    if (isAssembled()) {
        return;
    }

    // Assembly can be called only after adding all the rows
    if (countMissingRows() != 0) {
        throw std::runtime_error("Assembly can be called only after adding all the rows.");
    }

#if BITPIT_ENABLE_MPI==1
    // Updathe global information of the non-zero elements
    if (m_partitioned) {
        MPI_Allreduce(&m_maxRowNZ, &m_global_maxRowNZ, 1, MPI_LONG, MPI_MAX, m_communicator);
        MPI_Allreduce(&m_nNZ, &m_global_nNZ, 1, MPI_LONG, MPI_SUM, m_communicator);
    }
#endif

    // The function is now assembled
    m_assembled = true;
}

/**
* Check if the matrix is assembled and ready for use.

* \result Returns true if the matrix is assembled and ready for use.
*/
bool SparseMatrix::isAssembled() const
{
    return m_assembled;
}

/**
* Count the number of rows that needs to be added to fill the matrix.
*
* \result The number of rows that needs to be added to fill the matrix.
*/
long SparseMatrix::countMissingRows() const
{
    long nRows      = getRowCount();
    long nAddedRows = countAddedRows();

    return (nRows - nAddedRows);
}

/**
* Count the number of rows that have been added to the matrix.
*
* \result The number of rows that have been added to the matrix.
*/
long SparseMatrix::countAddedRows() const
{
    return (m_lastRow + 1);
}

/**
* Get the number of blocks of the matrix.
*
* \result The number of blocks of the matrix.
*/
int SparseMatrix::getBlockSize() const
{
    return m_blockSize;
}

/**
* Get the number of rows of the matrix.
*
* \result The number of rows of the matrix.
*/
long SparseMatrix::getRowCount() const
{
    return m_nRows;
}

/**
* Get the number of columns of the matrix.
*
* \result The number of columns of the matrix.
*/
long SparseMatrix::getColCount() const
{
    return m_nCols;
}

/**
* Get the number of rows of the matrix.
*
* \result The number of rows of the matrix.
*/
long SparseMatrix::getRowElementCount() const
{
    long nElements = getBlockSize() * getRowCount();

    return nElements;
}

/**
* Get the number of columns of the matrix.
*
* \result The number of columns of the matrix.
*/
long SparseMatrix::getColElementCount() const
{
    long nElements = getBlockSize() * getColCount();

    return nElements;
}

/**
* Get the number of non-zero elements.
*
* \result The number of non-zero elements.
*/
long SparseMatrix::getNZCount() const
{
    return m_nNZ;
}

/**
* Get the number of non-zero elements in the specified row.
*
* \result The number of non-zero elements in the specified row.
*/
long SparseMatrix::getRowNZCount(long row) const
{
    return m_pattern.getItemCount(row);
}

/**
* Get the maximum number of non-zero elements per row.
*
* \result The maximum number of non-zero elements per row.
*/
long SparseMatrix::getMaxRowNZCount() const
{
    return m_maxRowNZ;
}

/**
* Get the number of non-zero elements.
*
* \result The number of non-zero elements.
*/
long SparseMatrix::getNZElementCount() const
{
    long nElements = getBlockSize() * getNZCount();

    return nElements;
}

/**
* Get the number of non-zero elements in the specified row.
*
* \result The number of non-zero elements in the specified row.
*/
long SparseMatrix::getRowNZElementCount(long row) const
{
    long nElements = getBlockSize() * getRowNZCount(row);

    return nElements;
}

/**
* Get the maximum number of non-zero elements per row.
*
* \result The maximum number of non-zero elements per row.
*/
long SparseMatrix::getMaxRowNZElementCount() const
{
    long nElements = getBlockSize() * getMaxRowNZCount();

    return nElements;
}

#if BITPIT_ENABLE_MPI==1
/*!
	Checks if the matrix is partitioned.

	\result Returns true if the patch is partitioned, false otherwise.
*/
bool SparseMatrix::isPartitioned() const
{
	return m_partitioned;
}

/*!
	Gets the MPI communicator associated to the matrix.

	\return The MPI communicator associated to the matrix.
*/
const MPI_Comm & SparseMatrix::getCommunicator() const
{
	return m_communicator;
}

/*!
	Sets the MPI communicator to be used for parallel communications.

	\param communicator is the communicator to be used for parallel
	communications.
*/
void SparseMatrix::setCommunicator(MPI_Comm communicator)
{
    if ((communicator != MPI_COMM_NULL) && (communicator != MPI_COMM_SELF)) {
        MPI_Comm_dup(communicator, &m_communicator);
    } else {
        m_communicator = MPI_COMM_SELF;
    }
}

/*!
	Frees the MPI communicator associated to the matrix.
*/
void SparseMatrix::freeCommunicator()
{
    if (m_communicator != MPI_COMM_SELF) {
        int finalizedCalled;
        MPI_Finalized(&finalizedCalled);
        if (!finalizedCalled) {
            MPI_Comm_free(&m_communicator);
        }
    }
}

/**
* Get the number of global rows
*
* \result The number of global rows
*/
long SparseMatrix::getRowGlobalCount() const
{
    return m_global_nRows;
}

/**
* Get global row offset.
*
* \result The global row offset.
*/
long SparseMatrix::getRowGlobalOffset() const
{
    return m_global_rowOffset;
}

/**
* Get the number of global rows
*
* \result The number of global rows
*/
long SparseMatrix::getRowGlobalElementCount() const
{
    long nElements = getBlockSize() * getRowGlobalCount();

    return nElements;
}

/**
* Get global row offset.
*
* \result The global row offset.
*/
long SparseMatrix::getRowGlobalElementOffset() const
{
    long offset = getBlockSize() * getRowGlobalOffset();

    return offset;
}

/**
* Get number of global columns.
*
* \result The number of global columns.
*/
long SparseMatrix::getColGlobalCount() const
{
    return m_global_nCols;
}

/**
* Get global column offset.
*
* \result The global column offset.
*/
long SparseMatrix::getColGlobalOffset() const
{
    return m_global_colOffset;
}

/**
* Get the number of global column
*
* \result The number of global column
*/
long SparseMatrix::getColGlobalElementCount() const
{
    long nElements = getBlockSize() * getColGlobalCount();

    return nElements;
}

/**
* Get global column offset.
*
* \result The global column offset.
*/
long SparseMatrix::getColGlobalElementOffset() const
{
    long offset = getBlockSize() * getColGlobalOffset();

    return offset;
}

/**
* Get the global number of non-zero elements.
*
* \result The global number of non-zero elements.
*/
long SparseMatrix::getNZGlobalCount() const
{
    return m_global_nNZ;
}

/**
* Get the global maximum number of non-zero elements per row.
*
* \result The global maximum number of non-zero elements per row.
*/
long SparseMatrix::getMaxRowNZGlobalCount() const
{
    return m_global_maxRowNZ;
}

/**
* Get the global number of non-zero elements.
*
* \result The global number of non-zero elements.
*/
long SparseMatrix::getNZGlobalElementCount() const
{
    long nElement = getBlockSize() * getNZGlobalCount();

    return nElement;
}

/**
* Get the global maximum number of non-zero elements per row.
*
* \result The global maximum number of non-zero elements per row.
*/
long SparseMatrix::getMaxRowNZGlobalElementCount() const
{
    long nElement = getBlockSize() * getMaxRowNZGlobalCount();

    return nElement;
}

/**
* Extract the list of local global rows.
*
* \result The the list of local global rows.
*/
std::vector<long> SparseMatrix::extractLocalGlobalRows() const
{
    const std::size_t *rowExtents = m_pattern.indices();

    std::vector<long> localGlobalRows;
    localGlobalRows.reserve(m_nRows);
    for (long i = 0; i < m_nRows; ++i) {
        std::size_t nRowNZ = rowExtents[i + 1] - rowExtents[i];
        if (nRowNZ > 0) {
            localGlobalRows.push_back(m_global_rowOffset + i);
        }
    }

    return localGlobalRows;
}

/**
* Extract the list of ghost global rows.
*
* \result The the list of ghost global rows.
*/
std::vector<long> SparseMatrix::extractGhostGlobalRows() const
{
    return std::vector<long>();
}

/**
* Extract the list of local global columns.
*
* \result The the list of local global columns.
*/
std::vector<long> SparseMatrix::extractLocalGlobalCols() const
{
    long firstGlobalCol = m_global_colOffset;
    long lastGlobalCol  = m_global_colOffset + m_nCols - 1;

    const long *globalCols = m_pattern.data();

    std::vector<long> localGlobalCols;
    localGlobalCols.reserve(m_nNZ);
    for (long k = 0; k < m_nNZ; ++k) {
        long globalCol = globalCols[k];
        if (globalCol < firstGlobalCol || globalCol >= lastGlobalCol) {
            utils::addToOrderedVector<long>(globalCol, localGlobalCols);
        }
    }

    return localGlobalCols;
}

/**
* Extract the list of ghost global columns.
*
* \result The the list of ghost global columns.
*/
std::vector<long> SparseMatrix::extractGhostGlobalCols() const
{
    long firstGlobalCol = m_global_colOffset;
    long lastGlobalCol  = m_global_colOffset + m_nCols - 1;

    const long *globalCols = m_pattern.data();

    std::vector<long> ghostGlobalCols;
    if (m_nNZ > 0) {
        ghostGlobalCols.reserve(m_nNZ);
        for (long k = 0; k < m_nNZ; ++k) {
            long globalCol = globalCols[k];
            if (globalCol >= firstGlobalCol || globalCol < lastGlobalCol) {
                utils::addToOrderedVector<long>(globalCol, ghostGlobalCols);
            }
        }
    }

    return ghostGlobalCols;
}
#endif

/**
* Add a row.
*
* \param rowPattern are the indexes of the non-zero columns of the matrix
* \param rowValues are the values of the non-zero columns of the matrix
*/
void SparseMatrix::addRow(const std::vector<long> &rowPattern, const std::vector<double> &rowValues)
{
    addRow(rowPattern.size(), rowPattern.data(), rowValues.data());
}

/**
* Add a row.
*
* \param nRowNZ is the number of non-zero elements in the row
* \param rowPattern are the indexes of the non-zero columns of the matrix
* \param rowValues are the values of the non-zero columns of the matrix
*/
void SparseMatrix::addRow(long nRowNZ, const long *rowPattern, const double *rowValues)
{
    if (countMissingRows() == 0) {
        throw std::runtime_error("Unable to add another row: all rows have already been defined.");
    }

    // Add the row pattern
    m_pattern.pushBack(nRowNZ, rowPattern);

    // Add row values
    const int blockSize = getBlockSize();
    const int nBlockElements = blockSize * blockSize;

    m_values.insert(m_values.end(), rowValues, rowValues + nBlockElements * nRowNZ);

    // Update the non-zero element counters
    m_maxRowNZ = std::max(nRowNZ, m_maxRowNZ);
    m_nNZ += nRowNZ;

    // Update the index of the last row
    m_lastRow++;
}

/**
* Get the pattern of the specified row.
*
* \param row is the row
* \result The pattern of the specified row.
*/
ConstProxyVector<long> SparseMatrix::getRowPattern(long row) const
{
    ConstProxyVector<long> pattern;
    getRowPattern(row, &pattern);

    return pattern;
}

/**
* Get the pattern of the specified row.
*
* \param row is the row
* \param pattern on output will contain the pattern of the specified row
*/
void SparseMatrix::getRowPattern(long row, ConstProxyVector<long> *pattern) const
{
    const std::size_t rowPatternSize  = getRowNZCount(row);
    const long *rowPattern = getRowPatternData(row);
    pattern->set(rowPattern, rowPatternSize);
}

/**
* Get the values of the specified row.
*
* \param row is the row
* \result The values of the specified row.
*/
ConstProxyVector<double> SparseMatrix::getRowValues(long row) const
{
    ConstProxyVector<double> values;
    getRowValues(row, &values);

    return values;
}

/**
* Get the values of the specified row.
*
* \param row is the row
* \param values on output will contain the values of the specified row
*/
void SparseMatrix::getRowValues(long row, ConstProxyVector<double> *values) const
{
    const int blockSize = getBlockSize();
    const int nBlockElements = blockSize * blockSize;

    const std::size_t nRowValues = nBlockElements * getRowNZCount(row);
    const double *rowValues = getRowValuesData(row);

    values->set(rowValues, nRowValues);
}

/**
* Get a constant pointer to the internal pattern of the specified row.
*
* \param row is the row
* \result A constant pointer to the internal pattern of the specified row.
*/
long * SparseMatrix::getRowPatternData(long row)
{
    return const_cast<long *>(const_cast<const SparseMatrix *>(this)->getRowPatternData(row));
}

/**
* Get a constant pointer to the internal pattern of the specified row.
*
* \param row is the row
* \result A constant pointer to the internal pattern of the specified row.
*/
const long * SparseMatrix::getRowPatternData(long row) const
{
    const std::size_t *rowExtent      = m_pattern.indices(row);
    const std::size_t rowPatternBegin = rowExtent[0];

    const long *rowPattern = m_pattern.data() + rowPatternBegin;

    return rowPattern;
}

/**
* Get a constant pointer to the internal values of the specified row.
*
* \param row is the row
* \result A constant pointer to the internal values of the specified row.
*/
double * SparseMatrix::getRowValuesData(long row)
{
    return const_cast<double *>(const_cast<const SparseMatrix *>(this)->getRowValuesData(row));
}

/**
* Get a constant pointer to the internal values of the specified row.
*
* \param row is the row
* \result A constant pointer to the internal values of the specified row.
*/
const double * SparseMatrix::getRowValuesData(long row) const
{
    const int blockSize = getBlockSize();
    const int nBlockElements = blockSize * blockSize;

    const std::size_t *rowExtent  = m_pattern.indices(row);
    const std::size_t valuesBegin = nBlockElements * rowExtent[0];

    const double *rowValues = m_values.data() + valuesBegin;

    return rowValues;
}

/**
* Compute the transpose of the matrix.
*
* The transpose is evaluated as a new matrix, the current matrix is not
* modified.
*
* \result The transpose of the matrix.
*/
std::unique_ptr<SparseMatrix> SparseMatrix::computeTranspose() const
{
    // Early return if the matrix is not assembled.
    if (!isAssembled()) {
        return nullptr;
    }

    // Create the transpose matrix
    std::unique_ptr<SparseMatrix> transpose;
#if BITPIT_ENABLE_MPI==1
    transpose = std::unique_ptr<SparseMatrix>(new SparseMatrix(m_communicator, m_partitioned, m_nCols, m_nRows, m_nNZ));
#else
    transpose = std::unique_ptr<SparseMatrix>(new SparseMatrix(m_nCols, m_nRows, m_nNZ));
#endif

    // Create an empty transpose pattern
    std::vector<std::size_t> transposeRowSizes(transpose->m_nRows, 0.);
    for (int i = 0; i < m_nNZ; ++i) {
        long column = *(m_pattern.data() + i);
        ++transposeRowSizes[column];
    }

    transpose->m_pattern.initialize(transpose->m_nRows, transposeRowSizes.data(), 0.);

    // Create the empty storage for the values
    transpose->m_values.resize(m_blockSize * m_nNZ);

    // Set non-zero information
    transpose->m_nNZ = m_nNZ;

    transpose->m_maxRowNZ = 0;
    for (int i = 0; i < transpose->m_nRows; ++i) {
        transpose->m_maxRowNZ = std::max(static_cast<long>(transposeRowSizes[i]), transpose->m_maxRowNZ);
    }

    // Fill patter and values of the transpose matrix
    const std::size_t *rowExtents = m_pattern.indices();
    std::fill(transposeRowSizes.begin(), transposeRowSizes.end(), 0);
    for (long row = 0; row < m_nRows; ++row) {
        const std::size_t rowPatternSize = rowExtents[row + 1] - rowExtents[row];
        const long *rowPattern = m_pattern.data() + rowExtents[row];
        const double *rowValues = m_values.data() + rowExtents[row];
        for (std::size_t k = 0; k < rowPatternSize; ++k) {
            long column = rowPattern[k];
            const std::size_t transposeRowLastIndex = *(transpose->m_pattern.indices(column)) + transposeRowSizes[column];

            long *transposeRowLastPattern = transpose->m_pattern.data() + transposeRowLastIndex;
            *transposeRowLastPattern = row;

            double *transposeRowLastValue = transpose->m_values.data() + transposeRowLastIndex;
            *transposeRowLastValue = rowValues[k];

            ++transposeRowSizes[column];
        }
    }
    transpose->m_lastRow = transpose->m_nRows - 1;

    // Assembly the transpose matrix
    transpose->assembly();

    return transpose;
}

}
