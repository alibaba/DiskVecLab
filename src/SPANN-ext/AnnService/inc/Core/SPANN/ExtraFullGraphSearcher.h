// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASEARCHER_H_
#define _SPTAG_SPANN_EXTRASEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "inc/Core/Common/TruthSet.h"
#include "Compressor.h"

#include <map>
#include <cmath>
#include <climits>
#include <future>
#include <numeric>

namespace SPTAG
{
    namespace SPANN
    {
        extern std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO;

        struct Selection {
            std::string m_tmpfile;
            size_t m_totalsize;
            size_t m_start;
            size_t m_end;
            std::vector<Edge> m_selections;
            static EdgeCompare g_edgeComparer;

            Selection(size_t totalsize, std::string tmpdir) : m_tmpfile(tmpdir + FolderSep + "selection_tmp"), m_totalsize(totalsize), m_start(0), m_end(totalsize) { remove(m_tmpfile.c_str()); m_selections.resize(totalsize); }

            ErrorCode SaveBatch()
            {
                auto f_out = f_createIO();
                if (f_out == nullptr || !f_out->Initialize(m_tmpfile.c_str(), std::ios::out | std::ios::binary | (fileexists(m_tmpfile.c_str()) ? std::ios::in : 0))) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to save selection for batching!\n", m_tmpfile.c_str());
                    return ErrorCode::FailedOpenFile;
                }
                if (f_out->WriteBinary(sizeof(Edge) * (m_end - m_start), (const char*)m_selections.data(), sizeof(Edge) * m_start) != sizeof(Edge) * (m_end - m_start)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot write to %s!\n", m_tmpfile.c_str());
                    return ErrorCode::DiskIOFail;
                }
                std::vector<Edge> batch_selection;
                m_selections.swap(batch_selection);
                m_start = m_end = 0;
                return ErrorCode::Success;
            }

            ErrorCode LoadBatch(size_t start, size_t end)
            {
                auto f_in = f_createIO();
                if (f_in == nullptr || !f_in->Initialize(m_tmpfile.c_str(), std::ios::in | std::ios::binary)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to load selection batch!\n", m_tmpfile.c_str());
                    return ErrorCode::FailedOpenFile;
                }

                size_t readsize = end - start;
                m_selections.resize(readsize);
                if (f_in->ReadBinary(readsize * sizeof(Edge), (char*)m_selections.data(), start * sizeof(Edge)) != readsize * sizeof(Edge)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot read from %s! start:%zu size:%zu\n", m_tmpfile.c_str(), start, readsize);
                    return ErrorCode::DiskIOFail;
                }
                m_start = start;
                m_end = end;
                return ErrorCode::Success;
            }

            size_t lower_bound(SizeType node)
            {
                auto ptr = std::lower_bound(m_selections.begin(), m_selections.end(), node, g_edgeComparer);
                return m_start + (ptr - m_selections.begin());
            }

            Edge& operator[](size_t offset)
            {
                if (offset < m_start || offset >= m_end) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error read offset in selections:%zu\n", offset);
                }
                return m_selections[offset - m_start];
            }
        };

#define DecompressPosting(){\
        p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer(); \
        if (listInfo->listEleCount != 0) { \
            std::size_t sizePostingListFullData;\
            try {\
                sizePostingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);\
            }\
            catch (std::runtime_error& err) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", listInfo - m_listInfos.data(), err.what());\
                return;\
            }\
            if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", listInfo - m_listInfos.data(), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
                return;\
            }\
        }\
}\

#define DecompressPostingIterative(){\
        p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer(); \
        if (listInfo->listEleCount != 0) { \
            std::size_t sizePostingListFullData;\
            try {\
                sizePostingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);\
                if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", listInfo - m_listInfos.data(), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
                }\
             }\
            catch (std::runtime_error& err) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", listInfo - m_listInfos.data(), err.what());\
            }\
        }\
}\

#define ProcessPosting() \
        for (int i = 0; i < listInfo->listEleCount; i++) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);\
            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID));\
            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue; \
            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = p_index->ComputeDistanceQuantizer(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
        } \

#define ProcessPostingOffset() \
        while (p_exWorkSpace->m_offset < listInfo->listEleCount) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, p_exWorkSpace->m_offset, listInfo->listEleCount);\
            p_exWorkSpace->m_offset++;\
            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID));\
            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue; \
            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
            foundResult = true;\
            break;\
        } \
        if (p_exWorkSpace->m_offset == listInfo->listEleCount) { \
            p_exWorkSpace->m_pi++; \
            p_exWorkSpace->m_offset = 0; \
        } \

        template <typename ValueType>
        class ExtraFullGraphSearcher : public IExtraSearcher
        {
        public:
            ExtraFullGraphSearcher()
            {
                m_enableDeltaEncoding = false;
                m_enablePostingListRearrange = false;
                m_enableDataCompression = false;
                m_enableDictTraining = true;
            }

            virtual ~ExtraFullGraphSearcher()
            {
            }

            virtual bool LoadIndex(Options& p_opt) {
                m_extraFullGraphFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                std::string curFile = m_extraFullGraphFile;
                p_opt.m_searchPostingPageLimit = max(p_opt.m_searchPostingPageLimit, static_cast<int>((p_opt.m_postingVectorLimit * (p_opt.m_dim * sizeof(ValueType) + sizeof(int)) + PageSize - 1) / PageSize));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load index with posting page limit:%d\n", p_opt.m_searchPostingPageLimit);
                do {
                    auto curIndexFile = f_createAsyncIO();
                    if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in, 
#ifndef _MSC_VER
#ifdef BATCH_READ
                        p_opt.m_searchInternalResultNum, 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                        p_opt.m_searchInternalResultNum * p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
/*
#ifdef BATCH_READ
                        max(p_opt.m_searchInternalResultNum*m_vectorInfoSize, 1 << 12), 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                        p_opt.m_searchInternalResultNum* p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
*/
#else
                        (p_opt.m_searchPostingPageLimit + 1) * PageSize, 2, 2, (std::uint16_t)p_opt.m_ioThreads
#endif
                    )) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                        return false;
                    }

                    m_indexFiles.emplace_back(curIndexFile);
                    try {
                        m_totalListCount += LoadingHeadInfo(curFile, p_opt.m_searchPostingPageLimit, m_listInfos);
                    } 
                    catch (std::exception& e)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error occurs when loading HeadInfo:%s\n", e.what());
                        return false;
                    }

                    curFile = m_extraFullGraphFile + "_" + std::to_string(m_indexFiles.size());
                } while (fileexists(curFile.c_str()));
                m_oneContext = (m_indexFiles.size() == 1);

                m_enableDeltaEncoding = p_opt.m_enableDeltaEncoding;
                m_enablePostingListRearrange = p_opt.m_enablePostingListRearrange;
                m_enableDataCompression = p_opt.m_enableDataCompression;
                m_enableDictTraining = p_opt.m_enableDictTraining;

                if (m_enablePostingListRearrange) m_parsePosting = &ExtraFullGraphSearcher<ValueType>::ParsePostingListRearrange;
                else m_parsePosting = &ExtraFullGraphSearcher<ValueType>::ParsePostingList;
                if (m_enableDeltaEncoding) m_parseEncoding = &ExtraFullGraphSearcher<ValueType>::ParseDeltaEncoding;
                else m_parseEncoding = &ExtraFullGraphSearcher<ValueType>::ParseEncoding;
                
                m_listPerFile = static_cast<int>((m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());

#ifndef _MSC_VER
                Helper::AIOTimeout.tv_nsec = p_opt.m_iotimeout * 1000;
#endif
                return true;
            }

            virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats,
                std::set<int>* truth, std::map<int, std::set<int>>* found)
            {
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
 
                int diskRead = 0;
                int diskIO = 0;
                int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext? 0: curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    diskRead += listInfo->listPageCount;
                    diskIO += 1;
                    listElements += listInfo->listEleCount;

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_buffer = buffer;
                    request.m_status = (fileid << 16) | p_exWorkSpace->m_spaceID;
                    request.m_payload = (void*)listInfo; 
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    request.m_callback = [&p_exWorkSpace, &queryResults, &p_index, &request, this](bool success)
                    {
                        char* buffer = request.m_buffer;
                        ListInfo* listInfo = (ListInfo*)(request.m_payload);

                        // decompress posting list
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            DecompressPosting();
                        }

                        ProcessPosting();
                    };
#else // async read
                    request.m_callback = [&p_exWorkSpace, &request](bool success)
                    {
                        p_exWorkSpace->m_processIocp.push(&request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                        unprocessed--;
                    }
#endif
#else // sync read
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        throw std::runtime_error("File read mismatch");
                    }
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
#endif
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
                }
#endif
#endif
                if (truth) {
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

                        ListInfo* listInfo = &(m_listInfos[curPostingID]);
                        char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer();
                            if (listInfo->listEleCount != 0)
                            {
                                try {
                                    m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);
                                }
                                catch (std::runtime_error& err) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", curPostingID, err.what());
                                    continue;
                                }
                            }
                        }

                        for (size_t i = 0; i < listInfo->listEleCount; ++i) {
                            uint64_t offsetVectorID = m_enablePostingListRearrange ? (m_vectorInfoSize - sizeof(int)) * listInfo->listEleCount + sizeof(int) * i : m_vectorInfoSize * i; \
                            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID)); \
                            if (truth && truth->count(vectorID)) (*found)[curPostingID].insert(vectorID);
                        }
                    }
                }

                if (p_stats) 
                {
                    p_stats->m_totalListElementsCount = listElements;
                    p_stats->m_diskIOCount = diskIO;
                    p_stats->m_diskAccessCount = diskRead;
                }
            }

            virtual void SearchIndexWithoutParsing(ExtraWorkSpace* p_exWorkSpace)
            {
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                int diskRead = 0;
                int diskIO = 0;
                int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext ? 0 : curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    diskRead += listInfo->listPageCount;
                    diskIO += 1;
                    listElements += listInfo->listEleCount;

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_buffer = buffer;
                    request.m_status = (fileid << 16) | p_exWorkSpace->m_spaceID;
                    request.m_payload = (void*)listInfo;
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    request.m_callback = [this](bool success)
                    {
                        //char* buffer = request.m_buffer;
                        //ListInfo* listInfo = (ListInfo*)(request.m_payload);

                        // decompress posting list
                        /*
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            DecompressPosting();
                        }

                        ProcessPosting();
                        */
                    };
#else // async read
                    request.m_callback = [&p_exWorkSpace, &request](bool success)
                    {
                        p_exWorkSpace->m_processIocp.push(&request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                        unprocessed--;
                    }
#endif
#else // sync read
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        throw std::runtime_error("File read mismatch");
                    }
                    // decompress posting list
                    /*
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
                    */
#endif
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    // decompress posting list
                    /*
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
                    */
                }
#endif
#endif
            }

            virtual bool SearchNextInPosting(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
		std::shared_ptr<VectorIndex>& p_index)
            {
                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
                bool foundResult = false;
                while (!foundResult && p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) {

                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[p_exWorkSpace->m_pi]).GetBuffer());
                    ListInfo* listInfo = static_cast<ListInfo*>(p_exWorkSpace->m_diskRequests[p_exWorkSpace->m_pi].m_payload);
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression && p_exWorkSpace->m_offset == 0)
                    {
                        DecompressPostingIterative();
                    }
                    ProcessPostingOffset();
                }
                return !(p_exWorkSpace->m_pi == p_exWorkSpace->m_postingIDs.size());
            }

            virtual bool SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace,
                 QueryResult& p_query,
		 std::shared_ptr<VectorIndex> p_index)
            {
                if (p_exWorkSpace->m_loadPosting) {
                    SearchIndexWithoutParsing(p_exWorkSpace);
                    p_exWorkSpace->m_pi = 0;
                    p_exWorkSpace->m_offset = 0;
                    p_exWorkSpace->m_loadPosting = false;
                }

                return SearchNextInPosting(p_exWorkSpace, p_query, p_index);
            }

            std::string GetPostingListFullData(
                Options& p_opt,
                int postingListId,
                size_t p_postingListSize,
                Selection &p_selections,
                std::shared_ptr<VectorSet> p_fullVectors,
                bool p_enableDeltaEncoding = false,
                bool p_enablePostingListRearrange = false,
                const ValueType *headVector = nullptr,
				bool finalStep = false)
            {
                bool run_spilt = p_opt.m_enableOutputDataSplit && finalStep;
                std::string postingListFullData("");
                std::string vectors("");
                std::string vectorIDs("");
                std::string original_vectors("");
                size_t selectIdx = p_selections.lower_bound(postingListId);
                // iterate over all the vectors in the posting list
                for (int i = 0; i < p_postingListSize; ++i)
                {
                    if (p_selections[selectIdx].node != postingListId)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH! node:%d offset:%zu\n", postingListId, selectIdx);
                        throw std::runtime_error("Selection ID mismatch");
                    }
                    std::string vectorID("");
                    std::string vector("");
                    std::string original_vector("");

                    int vid = p_selections[selectIdx++].tonode;
                    vectorID.append(reinterpret_cast<char *>(&vid), sizeof(int));

                    ValueType *p_vector = reinterpret_cast<ValueType *>(p_fullVectors->GetVector(vid));
                    if (p_enableDeltaEncoding)
                    {
                        DimensionType n = p_fullVectors->Dimension();
                        std::vector<ValueType> p_vector_delta(n);
                        for (auto j = 0; j < n; j++)
                        {
                            p_vector_delta[j] = p_vector[j] - headVector[j];
                        }
                        vector.append(reinterpret_cast<char *>(&p_vector_delta[0]), p_fullVectors->PerVectorDataSize());
                    }
                    else
                    {
                        vector.append(reinterpret_cast<char *>(p_vector), p_fullVectors->PerVectorDataSize());
                    }
                    if (run_spilt) {
                        original_vector.append(reinterpret_cast<char *>(p_vector), p_fullVectors->PerVectorDataSize());
                    }

                    vectorIDs += vectorID;
                    vectors += vector;
					if (run_spilt) {
						original_vectors += original_vector;
					}
                    if (!p_enablePostingListRearrange)
                    {
                        postingListFullData += (vectorID + vector);
                    }
                }
                if (run_spilt) {
					SizeType p_postingListSize_convert = p_postingListSize;
					DimensionType id_n = 1;
                    DimensionType n = p_fullVectors->Dimension();
                    // Check for potential overflow before multiplication
                    if (static_cast<uint64_t>(p_postingListSize) * p_fullVectors->PerVectorDataSize() > std::numeric_limits<size_t>::max()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Overflow detected in postingListFullSize calculation! p_postingListSize: %d, PerVectorDataSize: %zu\n", 
                                     p_postingListSize, p_fullVectors->PerVectorDataSize());
                        throw std::runtime_error("Overflow in postingListFullSize calculation");
                    }
                    size_t postingListFullSize = static_cast<size_t>(p_postingListSize) * p_fullVectors->PerVectorDataSize();
					SizeType ID_length = sizeof(int) * p_postingListSize;
                    std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
                    bool init = true;
                    
                    // Debug logging for file paths
                    std::string outputFile = p_opt.m_splitDirectory + FolderSep + "_subshard-" + std::to_string(postingListId) + ".bin";
                    std::string outputIDsFile = p_opt.m_splitDirectory + FolderSep + "_subshard-" + std::to_string(postingListId) + "_ids_uint32.bin";
                    
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Attempting to create output files for posting list %d:\n", postingListId);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "  Output file: %s\n", outputFile.c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "  Output IDs file: %s\n", outputIDsFile.c_str());
                    
                    if (output == nullptr || outputIDs == nullptr) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create DiskIO objects for output files!\n");
                        init = false;
                    } else if (!output->Initialize(outputFile.c_str(), std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to initialize output file: %s\n", outputFile.c_str());
                        init = false;
                    } else if (!outputIDs->Initialize(outputIDsFile.c_str(), std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to initialize output IDs file: %s\n", outputIDsFile.c_str());
                        init = false;
                    }
                    
                    if (!init) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n", outputFile.c_str(), outputIDsFile.c_str());
                    }
                    
                    if (init) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Writing p_postingListSize (%zu bytes) to output file\n", sizeof(p_postingListSize_convert));
                        if (output->WriteBinary(sizeof(p_postingListSize_convert), reinterpret_cast<char*>(&p_postingListSize_convert)) != sizeof(p_postingListSize_convert)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file for p_postingListSize! Expected: %zu, Actual: %lld\n",
                                sizeof(p_postingListSize_convert), 
                                output->WriteBinary(sizeof(p_postingListSize_convert), reinterpret_cast<char*>(&p_postingListSize_convert)));
                            exit(1);
                        }
                        
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Writing dimension n (%zu bytes) to output file\n", sizeof(n));
                        if (output->WriteBinary(sizeof(n), reinterpret_cast<char*>(&n)) != sizeof(n)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file for n! Expected: %zu, Actual: %lld\n",
                                sizeof(n), 
                                output->WriteBinary(sizeof(n), reinterpret_cast<char*>(&n)));
                            exit(1);
                        }
                        
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Writing original_vectors (%zu bytes) to output file\n", postingListFullSize);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "original_vectors.data() pointer: %p\n", original_vectors.data());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "original_vectors size: %zu\n", original_vectors.size());
                        
                        auto writeResult = output->WriteBinary(postingListFullSize, const_cast<char *>(original_vectors.data()));
                        if (writeResult != postingListFullSize) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file for original_vectors! Expected: %zu, Actual: %lld\n",
                                postingListFullSize, writeResult);
                            // Additional debug info
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Posting list ID: %d\n", postingListId);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Posting list size: %d\n", p_postingListSize);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Per vector data size: %zu\n", p_fullVectors->PerVectorDataSize());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Total expected size: %zu\n", postingListFullSize);
                            exit(1);
                        }
                        
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Writing p_postingListSize to outputIDs file (%zu bytes)\n", sizeof(p_postingListSize_convert));
                        if (outputIDs->WriteBinary(sizeof(p_postingListSize_convert), reinterpret_cast<char*>(&p_postingListSize_convert)) != sizeof(p_postingListSize_convert)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputIDs file for p_postingListSize! Expected: %zu, Actual: %lld\n",
                                sizeof(p_postingListSize_convert), 
                                outputIDs->WriteBinary(sizeof(p_postingListSize_convert), reinterpret_cast<char*>(&p_postingListSize_convert)));
                            exit(1);
                        }
                        
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Writing id_n to outputIDs file (%zu bytes)\n", sizeof(id_n));
                        if (outputIDs->WriteBinary(sizeof(id_n), reinterpret_cast<char*>(&id_n)) != sizeof(id_n)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputIDs file for n! Expected: %zu, Actual: %lld\n",
                                sizeof(id_n), 
                                outputIDs->WriteBinary(sizeof(id_n), reinterpret_cast<char*>(&id_n)));
                            exit(1);
                        }
                        
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Writing vectorIDs (%zu bytes) to outputIDs file\n", ID_length);
                        auto idsWriteResult = outputIDs->WriteBinary(ID_length, const_cast<char *>(vectorIDs.data()));
                        if (idsWriteResult != ID_length) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputIDs file for vectorIDs! Expected: %zu, Actual: %lld\n",
                                ID_length, idsWriteResult);
                            exit(1);
                        }
                        
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Successfully wrote all data for posting list %d\n", postingListId);
                    }
                }
                if (p_enablePostingListRearrange)
                {
                    return vectors + vectorIDs;
                }
                return postingListFullData;
            }

            bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<Helper::VectorSetReader>& p_reader_quantizer,
                            std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt) {
                std::string outputFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                if (outputFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Output file can't be empty!\n");
                    return false;
                }

                int numThreads = p_opt.m_iSSDNumberOfThreads;
                int candidateNum = p_opt.m_internalResultNum;
                std::unordered_set<SizeType> headVectorIDS;
                if (p_opt.m_headIDFile.empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                    return false;
                }

                {
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize((p_opt.m_indexDirectory + FolderSep +  p_opt.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "failed open VectorIDTranslate: %s\n", p_opt.m_headIDFile.c_str());
                        return false;
                    }

                    std::uint64_t vid;
                    while (ptr->ReadBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) == sizeof(vid))
                    {
                        headVectorIDS.insert(static_cast<SizeType>(vid));
                        //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "头向量 vid: %u\n", vid);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded %u Vector IDs\n", static_cast<uint32_t>(headVectorIDS.size()));
                }

                SizeType fullCount = 0;
                size_t vectorInfoSize = 0;
                {
                    auto fullVectors = p_reader->GetVectorSet();
                    fullCount = fullVectors->Count();
                    vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);        // vecotorInfoSize 是按照全精度向量计算的
                }

                // 添加头向量统计信息
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== 头向量统计信息 ===\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "总向量数: %d\n", fullCount);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "头向量数: %u\n", static_cast<uint32_t>(headVectorIDS.size()));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "非头向量数(将进入posting lists): %d\n", fullCount - static_cast<int>(headVectorIDS.size()));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "头向量比例: %.2f%%\n", headVectorIDS.size() * 100.0f / fullCount);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "========================\n");

                Selection selections(static_cast<size_t>(fullCount) * p_opt.m_replicaCount, p_opt.m_tmpdir);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
                std::vector<std::atomic_int> replicaCount(fullCount);
                std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
                for (auto& pls : postingListSize) pls = 0;
                std::unordered_set<SizeType> emptySet;
                SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (p_opt.m_batches > 1)
                {
                    if (selections.SaveBatch() != ErrorCode::Success)
                    {
                        return false;
                    }
                }
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                    SizeType sampleSize = p_opt.m_samples;
                    std::vector<SizeType> samples(sampleSize, 0);
                    // 分批次处理 (start, end) 中的向量
                    for (int i = 0; i < p_opt.m_batches; i++) {
                        SizeType start = i * batchSize;
                        SizeType end = min(start + batchSize, fullCount);
                        auto fullVectors = p_reader->GetVectorSet(start, end);
                        if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                        if (p_opt.m_batches > 1) {
                            if (selections.LoadBatch(static_cast<size_t>(start) * p_opt.m_replicaCount, static_cast<size_t>(end) * p_opt.m_replicaCount) != ErrorCode::Success)
                            {
                                return false;
                            }
                            emptySet.clear();
                            for (auto vid : headVectorIDS) {
                                if (vid >= start && vid < end) emptySet.insert(vid - start);
                            }
                        }
                        else {
                            emptySet = headVectorIDS;
                        }

                        int sampleNum = 0;
                        for (int j = start; j < end && sampleNum < sampleSize; j++)
                        {
                            if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                        }

                        // 在这里面把 selection 赋值了
                        // full vectors 输入的input向量
                        p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                        // 处理每一个向量
                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                    // 数组：每个向量的多个副本挨着存储
                                    // .node 记录了这个点分给谁，所以把对应的posting list size加一下
                                    ++postingListSize[selections[vecOffset + resNum].node];
                                    // .tonode 记录了这个点是哪个向量的副本
                                    selections[vecOffset + resNum].tonode = j;
                                    ++replicaCount[j];
                                }
                            }
                        }
/*
                        // 添加：打印当前batch的posting list分配情况
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== Batch %d Posting List Assignments ===\n", i);
                        for (SizeType j = start; j < end; j++) { // 只打印前10个，避免输出过多
                            if (headVectorIDS.count(j) == 0) {
                                size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Vector[%d] -> PostingLists: ", j);
                                for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d ", selections[vecOffset + resNum].node);
                                }
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");
                            }
                        }
                        if (end - start > 10) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "... 共%d个向量，仅显示前10个\n", end - start);
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== Batch %d Assignment Complete ===\n", i);
*/

                        if (p_opt.m_batches > 1)
                        {
                            if (selections.SaveBatch() != ErrorCode::Success)
                            {
                                return false;
                            }
                        }
                    }
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

                // 添加：打印所有posting list的大小
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== 所有Posting List大小统计 ===\n");
                for (int i = 0; i < (int)postingListSize.size(); i++) { // 只打印前20个
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "PostingList[%d]: %d vectors\n", i, (int)postingListSize[i]);
                }
                if (postingListSize.size() > 20) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "... 共%zu个PostingList，仅显示前20个\n", postingListSize.size());
                }

                // 统计非空posting list数量
                int nonEmptyCount = 0;
                int totalVectors = 0;
                for (int i = 0; i < postingListSize.size(); i++) {
                    if (postingListSize[i] > 0) {
                        nonEmptyCount++;
                        totalVectors += postingListSize[i];
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "非空PostingList数量: %d/%zu\n", nonEmptyCount, postingListSize.size());
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "总向量数: %d, 平均每个PostingList: %.2f\n", totalVectors, totalVectors / (float)nonEmptyCount);

                // 添加：验证向量分配
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== 向量分配验证 ===\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "原始向量总数: %d\n", fullCount);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "头向量数量: %zu\n", headVectorIDS.size());
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "进入PostingList的向量数: %d\n", totalVectors);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "理论非头向量数: %d\n", fullCount - (int)headVectorIDS.size());
                if (totalVectors != fullCount - (int)headVectorIDS.size()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "警告：进入PostingList的向量数与理论值不符！差异: %d\n",
                        (fullCount - (int)headVectorIDS.size()) - totalVectors);
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== Posting List统计完成 ===\n");

                if (p_opt.m_batches > 1)
                {
                    if (selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount) != ErrorCode::Success)
                    {
                        return false;
                    }
                }

                // Sort results either in CPU or GPU
                VectorIndex::SortSelections(&selections.m_selections);

                auto t3 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

                int postingSizeLimit = INT_MAX;
                if (p_opt.m_postingPageLimit > 0)
                {
                    p_opt.m_postingPageLimit = max(p_opt.m_postingPageLimit, static_cast<int>((p_opt.m_postingVectorLimit * vectorInfoSize + PageSize - 1) / PageSize));
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build index with posting page limit:%d\n", p_opt.m_postingPageLimit);
                    postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);

                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0) continue;
                        ++replicaCountDist[replicaCount[i]];
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < postingListSize.size(); ++i)
                {
                    if (postingListSize[i] <= postingSizeLimit) continue;

                    std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, Selection::g_edgeComparer) - selections.m_selections.begin();

                    // 超限的直接drop，有减为 0 的风险
                    for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                    {
                        int tonode = selections.m_selections[selectIdx + dropID].tonode;
                        --replicaCount[tonode];
                    }
                    postingListSize[i] = postingSizeLimit;
                }

                if (p_opt.m_outputEmptyReplicaID)
                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                        return false;
                    }
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0) continue;

                        ++replicaCountDist[replicaCount[i]];

                        if (replicaCount[i] < 2)
                        {
                            long long vid = i;
                            if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
                                return false;
                            }
                        }
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

                auto t4 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

                // number of posting lists per file
                size_t postingFileSize = (postingListSize.size() + p_opt.m_ssdIndexFileNum - 1) / p_opt.m_ssdIndexFileNum;
                std::vector<size_t> selectionsBatchOffset(p_opt.m_ssdIndexFileNum + 1, 0);
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    selectionsBatchOffset[i + 1] = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), (SizeType)curPostingListEnd, Selection::g_edgeComparer) - selections.m_selections.begin();
                }

                if (p_opt.m_ssdIndexFileNum > 1)
                {
                    if (selections.SaveBatch() != ErrorCode::Success)
                    {
                        return false;
                    }
                }

                // 存储的是量化后的向量
                auto fullVectors = p_reader_quantizer->GetVectorSet();
                vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);
                if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                // iterate over files
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListOffSet = i * postingFileSize;
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    // postingListSize: number of vectors in the posting list, type vector<int>
                    std::vector<int> curPostingListSizes(
                        postingListSize.begin() + curPostingListOffSet,
                        postingListSize.begin() + curPostingListEnd);

                    std::vector<size_t> curPostingListBytes(curPostingListSizes.size());
                    
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Processing SSD index file %d/%d, posting lists: %zu-%zu\n", 
                        i+1, p_opt.m_ssdIndexFileNum, curPostingListOffSet, curPostingListEnd-1);
                    
                    if (p_opt.m_ssdIndexFileNum > 1)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Loading batch for SSD index file %d\n", i);
                        if (selections.LoadBatch(selectionsBatchOffset[i], selectionsBatchOffset[i + 1]) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to load batch for SSD index file %d\n", i);
                            return false;
                        }
                    }
                    // create compressor
                    if (p_opt.m_enableDataCompression && i == 0)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Creating compressor with level %d and buffer capacity %d\n", 
                            p_opt.m_zstdCompressLevel, p_opt.m_dictBufferCapacity);
                        m_pCompressor = std::make_unique<Compressor>(p_opt.m_zstdCompressLevel, p_opt.m_dictBufferCapacity);
                        // train dict
                        if (p_opt.m_enableDictTraining) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Training dictionary...\n");
                            std::string samplesBuffer("");
                            std::vector<size_t> samplesSizes;
                            for (int j = 0; j < curPostingListSizes.size(); j++) {
                                if (curPostingListSizes[j] == 0) {
                                    continue;
                                }
                                ValueType* headVector = nullptr;
                                if (p_opt.m_enableDeltaEncoding)
                                {
                                    headVector = (ValueType*)p_headIndex->GetSample(j);
                                }
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Getting posting list data for training dictionary, list %d\n", j);
                                std::string postingListFullData = GetPostingListFullData(p_opt,
                                    j, curPostingListSizes[j], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);

                                // 添加：打印posting list详细内容（仅前几个）
                                if (j < 3) { // 只打印前3个posting list的详细内容
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== 字典训练 PostingList[%d] 内容样本 ===\n", j);
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "包含向量数: %d, 数据大小: %zu bytes\n", curPostingListSizes[j], postingListFullData.size());

                                    // 从postingListFullData中解析向量ID（前几个）
                                    size_t offset = 0;
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "前5个向量ID: ");
                                    for (int k = 0; k < min(5, curPostingListSizes[j]); k++) {
                                        int vectorID = *(reinterpret_cast<const int*>(postingListFullData.data() + offset));
                                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d ", vectorID);
                                        offset += vectorInfoSize; // 跳到下一个向量
                                    }
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "=== PostingList[%d] 样本结束 ===\n", j);
                                }

                                samplesBuffer += postingListFullData;
                                samplesSizes.push_back(postingListFullData.size());
                                if (samplesBuffer.size() > p_opt.m_minDictTraingBufferSize) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Reached minimum training buffer size, stopping collection\n");
                                    break;
                                }
                            }
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using the first %zu postingLists to train dictionary... \n", samplesSizes.size());
                            std::size_t dictSize = m_pCompressor->TrainDict(samplesBuffer, &samplesSizes[0], (unsigned int)samplesSizes.size());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Dictionary trained, dictionary size: %zu \n", dictSize);
                        }
                    }

                    if (p_opt.m_enableDataCompression) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Getting compressed size of each posting list...\n");
#pragma omp parallel for schedule(dynamic)
                        for (int j = 0; j < curPostingListSizes.size(); j++) 
                        {
                            SizeType postingListId = j + (SizeType)curPostingListOffSet;
                            // do not compress if no data
                            if (postingListSize[postingListId] == 0) {
                                curPostingListBytes[j] = 0;
                                continue;
                            }
                            ValueType* headVector = nullptr;
                            if (p_opt.m_enableDeltaEncoding)
                            {
                                headVector = (ValueType*)p_headIndex->GetSample(postingListId);
                            }
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Getting posting list data for compression, list %d\n", postingListId);
                            std::string postingListFullData = GetPostingListFullData(p_opt,
                                postingListId, postingListSize[postingListId], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);
                            size_t sizeToCompress = postingListSize[postingListId] * vectorInfoSize;
                            if (sizeToCompress != postingListFullData.size()) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Size to compress NOT MATCH! PostingListFullData size: %zu sizeToCompress: %zu \n", postingListFullData.size(), sizeToCompress);
                            }
                            curPostingListBytes[j] = m_pCompressor->GetCompressedSize(postingListFullData, p_opt.m_enableDictTraining);
                            if (postingListId % 10000 == 0 || curPostingListBytes[j] > static_cast<uint64_t>(p_opt.m_postingPageLimit) * PageSize) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting list %d/%d, compressed size: %d, compression ratio: %.4f\n", postingListId, postingListSize.size(), curPostingListBytes[j], curPostingListBytes[j] / float(sizeToCompress));
                            }
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Got compressed size for all the %d posting lists in SSD Index file %d.\n", curPostingListBytes.size(), i);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mean compressed size: %.4f \n", std::accumulate(curPostingListBytes.begin(), curPostingListBytes.end(), 0.0) / curPostingListBytes.size());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mean compression ratio: %.4f \n", std::accumulate(curPostingListBytes.begin(), curPostingListBytes.end(), 0.0) / (std::accumulate(curPostingListSizes.begin(), curPostingListSizes.end(), 0.0) * vectorInfoSize));
                    }
                    else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Calculating uncompressed sizes for %zu posting lists\n", curPostingListSizes.size());
                        for (int j = 0; j < curPostingListSizes.size(); j++)
                        {
                            curPostingListBytes[j] = curPostingListSizes[j] * vectorInfoSize;
                        }
                    }

                    std::unique_ptr<int[]> postPageNum;
                    std::unique_ptr<std::uint16_t[]> postPageOffset;
                    std::vector<int> postingOrderInIndex;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Selecting posting offset for %zu posting lists\n", curPostingListBytes.size());
                    SelectPostingOffset(curPostingListBytes, postPageNum, postPageOffset, postingOrderInIndex);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Outputting SSD index file %d/%d\n", i+1, p_opt.m_ssdIndexFileNum);
                    OutputSSDIndexFile((i == 0) ? outputFile : outputFile + "_" + std::to_string(i),
                        p_opt.m_enableDeltaEncoding,
                        p_opt.m_enablePostingListRearrange,
                        p_opt.m_enableDataCompression,
                        p_opt.m_enableDictTraining,
                        p_opt,
                        vectorInfoSize,
                        curPostingListSizes,
                        curPostingListBytes,
                        p_headIndex,
                        selections,
                        postPageNum,
                        postPageOffset,
                        postingOrderInIndex,
                        fullVectors,
                        curPostingListOffSet);
                }

                auto t5 = std::chrono::high_resolution_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
             
                return true;
            }

            virtual bool CheckValidPosting(SizeType postingID)
            {
                return m_listInfos[postingID].listEleCount != 0;
            }


            virtual ErrorCode GetPostingDebug(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorIndex> p_index, SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs)
            {
                VIDs.clear();

                SizeType curPostingID = vid;
                ListInfo* listInfo = &(m_listInfos[curPostingID]);
                VIDs.resize(listInfo->listEleCount);
                ByteArray vector_array = ByteArray::Alloc(sizeof(ValueType) * listInfo->listEleCount * m_iDataDimension);
                vecs.reset(new BasicVectorSet(vector_array, GetEnumValueType<ValueType>(), m_iDataDimension, listInfo->listEleCount));

                int fileid = m_oneContext ? 0 : curPostingID / m_listPerFile;

#ifndef BATCH_READ
                Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[0]).GetBuffer());

#ifdef ASYNC_READ       
                auto& request = p_exWorkSpace->m_diskRequests[0];
                request.m_offset = listInfo->listOffset;
                request.m_readSize = totalBytes;
                request.m_buffer = buffer;
                request.m_status = (fileid << 16) | p_exWorkSpace->m_spaceID;
                request.m_payload = (void*)listInfo;
                request.m_success = false;

#ifdef BATCH_READ // async batch read
                request.m_callback = [&p_exWorkSpace, &vecs, &VIDs, &p_index, &request, this](bool success)
                {
                    char* buffer = request.m_buffer;
                    ListInfo* listInfo = (ListInfo*)(request.m_payload);

                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    for (int i = 0; i < listInfo->listEleCount; i++) 
                    {
                            uint64_t offsetVectorID, offsetVector; 
                            (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount); 
                            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID)); 
                            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector)); 
                            VIDs[i] = vectorID;
                            auto outVec = vecs->GetVector(i);
                            memcpy(outVec, (void*)(p_postingListFullData + offsetVector), sizeof(ValueType) * m_iDataDimension);
                    } 
                };
#else // async read
                request.m_callback = [&p_exWorkSpace, &request](bool success)
                {
                    p_exWorkSpace->m_processIocp.push(&request);
                };

                ++unprocessed;
                if (!(indexFile->ReadFileAsync(request)))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                    unprocessed--;
                }
#endif
#else // sync read
                auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                if (numRead != totalBytes) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                    throw std::runtime_error("File read mismatch");
                }
                // decompress posting list
                char* p_postingListFullData = buffer + listInfo->pageOffset;
                if (m_enableDataCompression)
                {
                    DecompressPosting();
                }

                for (int i = 0; i < listInfo->listEleCount; i++) 
                {
                    uint64_t offsetVectorID, offsetVector;
                    (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);
                    int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID));
                    (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));
                    VIDs[i] = vectorID;
                    auto outVec = vecs->GetVector(i);
                    memcpy(outVec, (void*)(p_postingListFullData + offsetVector), sizeof(ValueType) * m_iDataDimension);
                }
#endif
                return ErrorCode::Success;
            }

        private:
            struct ListInfo
            {
                std::size_t listTotalBytes = 0;
                
                int listEleCount = 0;

                std::uint16_t listPageCount = 0;

                std::uint64_t listOffset = 0;

                std::uint16_t pageOffset = 0;
            };

            int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, std::vector<ListInfo>& p_listInfos)
            {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                    throw std::runtime_error("Failed open file in LoadingHeadInfo");
                }
                m_pCompressor = std::make_unique<Compressor>(); // no need compress level to decompress

                int m_listCount;
                int m_totalDocumentCount;
                int m_listPageOffset;

                if (ptr->ReadBinary(sizeof(m_listCount), reinterpret_cast<char*>(&m_listCount)) != sizeof(m_listCount)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(m_totalDocumentCount), reinterpret_cast<char*>(&m_totalDocumentCount)) != sizeof(m_totalDocumentCount)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(m_iDataDimension), reinterpret_cast<char*>(&m_iDataDimension)) != sizeof(m_iDataDimension)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "m_iDataDimension = %d\n", m_iDataDimension);
                if (ptr->ReadBinary(sizeof(m_listPageOffset), reinterpret_cast<char*>(&m_listPageOffset)) != sizeof(m_listPageOffset)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }

                if (m_vectorInfoSize == 0) m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
                else if (m_vectorInfoSize != m_iDataDimension * sizeof(ValueType) + sizeof(int)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file! DataDimension and ValueType are not match!\n");
                    throw std::runtime_error("DataDimension and ValueType don't match in LoadingHeadInfo");
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "m_vectorInfoSize = %d\n", m_vectorInfoSize);

                size_t totalListCount = p_listInfos.size();
                p_listInfos.resize(totalListCount + m_listCount);

                size_t totalListElementCount = 0;

                std::map<int, int> pageCountDist;

                size_t biglistCount = 0;
                size_t biglistElementCount = 0;
                int pageNum;
                for (int i = 0; i < m_listCount; ++i)
                {
                    ListInfo* listInfo = &(p_listInfos[totalListCount + i]);

                    if (m_enableDataCompression)
                    {
                        if (ptr->ReadBinary(sizeof(listInfo->listTotalBytes), reinterpret_cast<char*>(&(listInfo->listTotalBytes))) != sizeof(listInfo->listTotalBytes)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                            throw std::runtime_error("Failed read file in LoadingHeadInfo");
                        }
                    }
                    if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&(pageNum))) != sizeof(pageNum)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->pageOffset), reinterpret_cast<char*>(&(listInfo->pageOffset))) != sizeof(listInfo->pageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->listEleCount), reinterpret_cast<char*>(&(listInfo->listEleCount))) != sizeof(listInfo->listEleCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->listPageCount), reinterpret_cast<char*>(&(listInfo->listPageCount))) != sizeof(listInfo->listPageCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    listInfo->listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
                    if (!m_enableDataCompression)
                    {
                        listInfo->listTotalBytes = listInfo->listEleCount * m_vectorInfoSize;
                        listInfo->listEleCount = min(listInfo->listEleCount, (min(static_cast<int>(listInfo->listPageCount), p_postingPageLimit) << PageSizeEx) / m_vectorInfoSize);
                        listInfo->listPageCount = static_cast<std::uint16_t>(ceil((m_vectorInfoSize * listInfo->listEleCount + listInfo->pageOffset) * 1.0 / (1 << PageSizeEx)));
                    }
                    totalListElementCount += listInfo->listEleCount;
                    int pageCount = listInfo->listPageCount;

                    if (pageCount > 1)
                    {
                        ++biglistCount;
                        biglistElementCount += listInfo->listEleCount;
                    }

                    if (pageCountDist.count(pageCount) == 0)
                    {
                        pageCountDist[pageCount] = 1;
                    }
                    else
                    {
                        pageCountDist[pageCount] += 1;
                    }
                }

                if (m_enableDataCompression && m_enableDictTraining)
                {
                    size_t dictBufferSize;
                    if (ptr->ReadBinary(sizeof(size_t), reinterpret_cast<char*>(&dictBufferSize)) != sizeof(dictBufferSize)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    char* dictBuffer = new char[dictBufferSize];
                    if (ptr->ReadBinary(dictBufferSize, dictBuffer) != dictBufferSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    try {
                        m_pCompressor->SetDictBuffer(std::string(dictBuffer, dictBufferSize));
                    }
                    catch (std::runtime_error& err) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file: %s \n", err.what());
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    delete[] dictBuffer;
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "Finish reading header info, list count %d, total doc count %d, dimension %d, list page offset %d.\n",
                    m_listCount,
                    m_totalDocumentCount,
                    m_iDataDimension,
                    m_listPageOffset);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "Big page (>4K): list count %zu, total element count %zu.\n",
                    biglistCount,
                    biglistElementCount);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n", totalListElementCount);

                for (auto& ele : pageCountDist)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Page Count Dist: %d %d\n", ele.first, ele.second);
                }

                return m_listCount;
            }

            inline void ParsePostingListRearrange(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                offsetVectorID = (m_vectorInfoSize - sizeof(int)) * eleCount + sizeof(int) * i;
                offsetVector = (m_vectorInfoSize - sizeof(int)) * i;
            }

            inline void ParsePostingList(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ParsePostingList");
                offsetVectorID = m_vectorInfoSize * i;
                offsetVector = offsetVectorID + sizeof(int);
            }

            inline void ParseDeltaEncoding(std::shared_ptr<VectorIndex>& p_index, ListInfo* p_info, ValueType* vector)
            {
                ValueType* headVector = (ValueType*)p_index->GetSample((SizeType)(p_info - m_listInfos.data()));
                COMMON::SIMDUtils::ComputeSum(vector, headVector, m_iDataDimension);
            }

            inline void ParseEncoding(std::shared_ptr<VectorIndex>& p_index, ListInfo* p_info, ValueType* vector) { //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ParseEncoding");
            }

            void SelectPostingOffset(
                const std::vector<size_t>& p_postingListBytes,
                std::unique_ptr<int[]>& p_postPageNum,
                std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                std::vector<int>& p_postingOrderInIndex)
            {
                p_postPageNum.reset(new int[p_postingListBytes.size()]);
                p_postPageOffset.reset(new std::uint16_t[p_postingListBytes.size()]);

                struct PageModWithID
                {
                    int id;

                    std::uint16_t rest;
                };

                struct PageModeWithIDCmp
                {
                    bool operator()(const PageModWithID& a, const PageModWithID& b) const
                    {
                        return a.rest == b.rest ? a.id < b.id : a.rest > b.rest;
                    }
                };

                std::set<PageModWithID, PageModeWithIDCmp> listRestSize;

                p_postingOrderInIndex.clear();
                p_postingOrderInIndex.reserve(p_postingListBytes.size());

                PageModWithID listInfo;
                for (size_t i = 0; i < p_postingListBytes.size(); ++i)
                {
                    if (p_postingListBytes[i] == 0)
                    {
                        continue;
                    }

                    listInfo.id = static_cast<int>(i);
                    listInfo.rest = static_cast<std::uint16_t>(p_postingListBytes[i] % PageSize);

                    listRestSize.insert(listInfo);
                }

                listInfo.id = -1;

                int currPageNum = 0;
                std::uint16_t currOffset = 0;

                while (!listRestSize.empty())
                {
                    listInfo.rest = PageSize - currOffset;
                    auto iter = listRestSize.lower_bound(listInfo); // avoid page-crossing
                    if (iter == listRestSize.end() || (listInfo.rest != PageSize && iter->rest == 0))
                    {
                        ++currPageNum;
                        currOffset = 0;
                    }
                    else
                    {
                        p_postPageNum[iter->id] = currPageNum;
                        p_postPageOffset[iter->id] = currOffset;

                        p_postingOrderInIndex.push_back(iter->id);

                        currOffset += iter->rest;
                        if (currOffset > PageSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Crossing extra pages\n");
                            throw std::runtime_error("Read too many pages");
                        }

                        if (currOffset == PageSize)
                        {
                            ++currPageNum;
                            currOffset = 0;
                        }

                        currPageNum += static_cast<int>(p_postingListBytes[iter->id] / PageSize);

                        listRestSize.erase(iter);
                    }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TotalPageNumbers: %d, IndexSize: %llu\n", currPageNum, static_cast<uint64_t>(currPageNum) * PageSize + currOffset);
            }

            void OutputSSDIndexFile(const std::string& p_outputFile,
                bool p_enableDeltaEncoding,
                bool p_enablePostingListRearrange,
                bool p_enableDataCompression,
                bool p_enableDictTraining,
                Options& p_opt,
                size_t p_spacePerVector,
                const std::vector<int>& p_postingListSizes,
                const std::vector<size_t>& p_postingListBytes,
                std::shared_ptr<VectorIndex> p_headIndex,
                Selection& p_postingSelections,
                const std::unique_ptr<int[]>& p_postPageNum,
                const std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                const std::vector<int>& p_postingOrderInIndex,
                std::shared_ptr<VectorSet> p_fullVectors,
                size_t p_postingListOffset)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                auto t1 = std::chrono::high_resolution_clock::now();

                auto ptr = SPTAG::f_createIO();
                int retry = 3;
                // open file 
                while (retry > 0 && (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open file %s, retrying...\n", p_outputFile.c_str());
                    retry--;
                }

                if (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open file %s\n", p_outputFile.c_str());
                    throw std::runtime_error("Failed to open file for SSD index save");
                }
                // meta size of global info
                std::uint64_t listOffset = sizeof(int) * 4;
                // meta size of the posting lists
                listOffset += (sizeof(int) + sizeof(std::uint16_t) + sizeof(int) + sizeof(std::uint16_t)) * p_postingListSizes.size();
                // write listTotalBytes only when enabled data compression
                if (p_enableDataCompression)
                {
                    listOffset += sizeof(size_t) * p_postingListSizes.size();
                }

                // compression dict
                if (p_enableDataCompression && p_enableDictTraining)
                {
                    listOffset += sizeof(size_t);
                    listOffset += m_pCompressor->GetDictBuffer().size();
                }

                std::unique_ptr<char[]> paddingVals(new char[PageSize]);
                memset(paddingVals.get(), 0, sizeof(char) * PageSize);
                // paddingSize: bytes left in the last page
                std::uint64_t paddingSize = PageSize - (listOffset % PageSize);
                if (paddingSize == PageSize)
                {
                    paddingSize = 0;
                }
                else
                {
                    listOffset += paddingSize;
                }

                // Number of posting lists
                int i32Val = static_cast<int>(p_postingListSizes.size());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Number of vectors
                i32Val = static_cast<int>(p_fullVectors->Count());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Vector dimension
                i32Val = static_cast<int>(p_fullVectors->Dimension());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "write m_iDataDimension = %d\n", i32Val);

                // Page offset of list content section
                i32Val = static_cast<int>(listOffset / PageSize);
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Meta of each posting list
                for (int i = 0; i < p_postingListSizes.size(); ++i)
                {
                    size_t postingListByte = 0;
                    int pageNum = 0; // starting page number
                    std::uint16_t pageOffset = 0;
                    int listEleCount = 0;
                    std::uint16_t listPageCount = 0;

                    if (p_postingListSizes[i] > 0)
                    {
                        pageNum = p_postPageNum[i];
                        pageOffset = static_cast<std::uint16_t>(p_postPageOffset[i]);
                        listEleCount = static_cast<int>(p_postingListSizes[i]);
                        postingListByte = p_postingListBytes[i];
                        listPageCount = static_cast<std::uint16_t>(postingListByte / PageSize);
                        if (0 != (postingListByte % PageSize))
                        {
                            ++listPageCount;
                        }
                    }
                    // Total bytes of the posting list, write only when enabled data compression
                    if (p_enableDataCompression && ptr->WriteBinary(sizeof(postingListByte), reinterpret_cast<char*>(&postingListByte)) != sizeof(postingListByte)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page number of the posting list
                    if (ptr->WriteBinary(sizeof(pageNum), reinterpret_cast<char*>(&pageNum)) != sizeof(pageNum)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page offset
                    if (ptr->WriteBinary(sizeof(pageOffset), reinterpret_cast<char*>(&pageOffset)) != sizeof(pageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Number of vectors in the posting list
                    if (ptr->WriteBinary(sizeof(listEleCount), reinterpret_cast<char*>(&listEleCount)) != sizeof(listEleCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page count of the posting list
                    if (ptr->WriteBinary(sizeof(listPageCount), reinterpret_cast<char*>(&listPageCount)) != sizeof(listPageCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }
                // compression dict
                if (p_enableDataCompression && p_enableDictTraining)
                {
                    std::string dictBuffer = m_pCompressor->GetDictBuffer();
                    // dict size
                    size_t dictBufferSize = dictBuffer.size();
                    if (ptr->WriteBinary(sizeof(size_t), reinterpret_cast<char *>(&dictBufferSize)) != sizeof(dictBufferSize))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // dict
                    if (ptr->WriteBinary(dictBuffer.size(), const_cast<char *>(dictBuffer.data())) != dictBuffer.size())
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                // Write padding vals
                if (paddingSize > 0)
                {
                    if (ptr->WriteBinary(paddingSize, reinterpret_cast<char*>(paddingVals.get())) != paddingSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                if (static_cast<uint64_t>(ptr->TellP()) != listOffset)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "List offset not match!\n");
                    throw std::runtime_error("List offset mismatch");
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SubIndex Size: %llu bytes, %llu MBytes\n", listOffset, listOffset >> 20);

                bool p_enableOutputDataSplit = p_opt.m_enableOutputDataSplit;
                auto head_output = SPTAG::f_createIO();
                DimensionType head_n = p_fullVectors->Dimension();
                SizeType head_size = p_fullVectors->PerVectorDataSize();
                SizeType head_postingListSize = 0;
                SizeType head_miss = 0;
                if (p_enableOutputDataSplit) {
					for (int i = 0; i < p_postingListSizes.size(); ++i) {
						if (p_postingListSizes[i] > 0) {
							head_postingListSize++;
						} else {
							head_miss++;
						}
					}
                    if (head_output == nullptr ||
                        !head_output->Initialize((p_opt.m_splitDirectory + FolderSep + "_centroids.bin").c_str(), std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s\n",
                        (p_opt.m_splitDirectory + FolderSep + "_centroids.bin").c_str());
                        exit(-1);
                        }
                    if (head_output->WriteBinary(sizeof(head_postingListSize), reinterpret_cast<char*>(&head_postingListSize)) != sizeof(head_postingListSize)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file for head_postingListSize!\n");
                        exit(1);
                    }
                    if (head_output->WriteBinary(sizeof(head_n), reinterpret_cast<char*>(&head_n)) != sizeof(head_n)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file for head_n!\n");
                        exit(1);
                    }
                }

                listOffset = 0;

                std::uint64_t paddedSize = 0;
                // iterate over all the posting lists
                for (auto id : p_postingOrderInIndex)
                {
                    std::uint64_t targetOffset = static_cast<uint64_t>(p_postPageNum[id]) * PageSize + p_postPageOffset[id];
                    if (targetOffset < listOffset)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "List offset not match, targetOffset < listOffset!\n");
                        throw std::runtime_error("List offset mismatch");
                    }
                    // write padding vals before the posting list
                    if (targetOffset > listOffset)
                    {
                        if (targetOffset - listOffset > PageSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Padding size greater than page size!\n");
                            throw std::runtime_error("Padding size mismatch with page size");
                        }

                        if (ptr->WriteBinary(targetOffset - listOffset, reinterpret_cast<char*>(paddingVals.get())) != targetOffset - listOffset) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }

                        paddedSize += targetOffset - listOffset;

                        listOffset = targetOffset;
                    }

                    if (p_postingListSizes[id] == 0)
                    {
                        continue;
                    }
                    int postingListId = id + (int)p_postingListOffset;
                    // get posting list full content and write it at once
                    ValueType *headVector = nullptr;
                    if (p_enableDeltaEncoding)
                    {
                        headVector = (ValueType *)p_headIndex->GetSample(postingListId);
                    }
                    if (p_enableOutputDataSplit) {
                        ValueType *originalHeadVector = (ValueType *)p_headIndex->GetSample(postingListId);
						if (postingListId < 3) {
							std::string msg = "postingListId=" + std::to_string(postingListId) + ", p_postingListSizes=" + std::to_string(p_postingListSizes.size()) +
							", p_postingOrderInIndex=" + std::to_string(p_postingOrderInIndex.size()) + ", head_postingListSize=" + std::to_string(head_postingListSize) +
							", head_miss=" + std::to_string(head_miss) + ", head_n=" + std::to_string(head_n) + ", head_size=" + std::to_string(head_size) + ", ValueType=" + std::to_string(sizeof(ValueType));
							SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%s\n", msg.c_str());
							for (int i = 0; i < 10; i++) {
								SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "==%f==,", originalHeadVector[i]);
							}
							SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%f,", originalHeadVector[127]);
							SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\n", msg.c_str());
						}
                        if (head_output->WriteBinary(head_size, reinterpret_cast<char*>(originalHeadVector)) != head_size) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write head File!");
                            throw std::runtime_error("Failed to write head File");
                        }
                    }
                    // Debug logging for large data handling
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Processing posting list %d (id: %d), size: %d\n", postingListId, id, p_postingListSizes[id]);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Memory usage before GetPostingListFullData for large posting list %d\n", postingListId);
                    
                    std::string postingListFullData = GetPostingListFullData(p_opt,
                        postingListId, p_postingListSizes[id], p_postingSelections, p_fullVectors, p_enableDeltaEncoding, p_enablePostingListRearrange, headVector, true);
                    
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Memory usage after GetPostingListFullData for large posting list %d\n", postingListId);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "postingListFullData.size(): %zu\n", postingListFullData.size());
                    
                    size_t postingListFullSize = p_postingListSizes[id] * p_spacePerVector;
                    if (postingListFullSize != postingListFullData.size())
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "posting list full data size NOT MATCH! postingListFullData.size(): %zu postingListFullSize: %zu \n", postingListFullData.size(), postingListFullSize);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Posting list ID: %d, posting list size: %d, space per vector: %zu\n", postingListId, p_postingListSizes[id], p_spacePerVector);
                        throw std::runtime_error("Posting list full size mismatch");
                    }
                    
                    if (p_enableDataCompression)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Compressing data for posting list %d, original size: %zu\n", postingListId, postingListFullData.size());
                        std::string compressedData = m_pCompressor->Compress(postingListFullData, p_enableDictTraining);
                        size_t compressedSize = compressedData.size();
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Compressed size for posting list %d: %zu\n", postingListId, compressedSize);
                        
                        if (compressedSize != p_postingListBytes[id])
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Compressed size NOT MATCH! compressed size:%zu, pre-calculated compressed size:%zu\n", compressedSize, p_postingListBytes[id]);
                            throw std::runtime_error("Compression size mismatch");
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Writing compressed data for posting list %d, size: %zu\n", postingListId, compressedSize);
                        if (ptr->WriteBinary(compressedSize, compressedData.data()) != compressedSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File for posting list %d! Expected: %zu, Actual: %lld\n", postingListId, compressedSize, ptr->WriteBinary(compressedSize, compressedData.data()));
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                        listOffset += compressedSize;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Successfully wrote compressed data for posting list %d\n", postingListId);
                    }
                    else
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Writing uncompressed data for posting list %d, size: %zu\n", postingListId, postingListFullSize);
                        auto writeResult = ptr->WriteBinary(postingListFullSize, postingListFullData.data());
                        if (writeResult != postingListFullSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File for posting list %d! Expected: %zu, Actual: %lld\n", postingListId, postingListFullSize, writeResult);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Posting list ID: %d, posting list size: %d\n", postingListId, p_postingListSizes[id]);
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                        listOffset += postingListFullSize;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "Successfully wrote uncompressed data for posting list %d\n", postingListId);
                    }
                }

                paddingSize = PageSize - (listOffset % PageSize);
                if (paddingSize == PageSize)
                {
                    paddingSize = 0;
                }
                else
                {
                    listOffset += paddingSize;
                    paddedSize += paddingSize;
                }

                if (paddingSize > 0)
                {
                    if (ptr->WriteBinary(paddingSize, reinterpret_cast<char *>(paddingVals.get())) != paddingSize)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Padded Size: %llu, final total size: %llu.\n", paddedSize, listOffset);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Output done...\n");
                auto t2 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to write results:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000);
            }

        private:
            
            std::string m_extraFullGraphFile;

            std::vector<ListInfo> m_listInfos;
            bool m_oneContext;

            std::vector<std::shared_ptr<Helper::DiskIO>> m_indexFiles;
            std::unique_ptr<Compressor> m_pCompressor;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableDataCompression;
            bool m_enableDictTraining;

            void (ExtraFullGraphSearcher<ValueType>::*m_parsePosting)(uint64_t&, uint64_t&, int, int);
            void (ExtraFullGraphSearcher<ValueType>::*m_parseEncoding)(std::shared_ptr<VectorIndex>&, ListInfo*, ValueType*);

            int m_vectorInfoSize = 0;
            int m_iDataDimension = 0;

            int m_totalListCount = 0;

            int m_listPerFile = 0;
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_EXTRASEARCHER_H_
