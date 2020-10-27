#include "search_server.h"
#include "iterator_range.h"
#include "parse.h"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <future>
using namespace std;

InvertedIndex::InvertedIndex(istream& document_input) {
  for (string current_doc; getline(document_input, current_doc);)
  {
    docs.push_back(move(current_doc)); // v
    size_t current_id = docs.size() - 1; // 0, 1;
    for (string_view sv: SplitIntoWords(docs.back()))
    {
      auto& now_stats = index[sv];
      if (!now_stats.empty() && now_stats.back().doc_id == current_id)
      {
        ++now_stats.back().hitcount;
      } else {
        now_stats.push_back({current_id, 1});
      }
    }
  }
}

const vector<InvertedIndex::Stats>& InvertedIndex::Lookup(string_view word) const
{
  static const vector<Stats> empty;
  if (auto it = index.find(word); it != index.end())
  {
    return it -> second;
  }
  else return empty;
}

void SearchServer::UpdateDocumentBase(istream& document_input) {
 tasks_.push_back(async([&] {
    InvertedIndex new_index(document_input);
    swap(index.GetAccess().ref_to_value, new_index);
    }));
}

void Do_Search(istream& query_input, ostream& search_results_output, Synchronized<InvertedIndex>& index_handler) 
{
  vector<size_t> docid_count;
  vector<int64_t> doc_ids;
  for (string current_query; getline(query_input, current_query);) 
  {
    const auto words = SplitIntoWords(current_query); 
    auto handle = index_handler.GetAccess();
    const size_t doc_count = handle.ref_to_value.GetDocuments().size();
    
    docid_count.assign(doc_count, 0);
    doc_ids.resize(doc_count);
    
    auto& index = handle.ref_to_value;
    for (const auto& word : words) {
      for (const auto&[docid, hit_count] : index.Lookup(word)) {
        docid_count[docid] += hit_count;
      }
    }
    iota(begin(doc_ids), end(doc_ids), 0);
    {
      partial_sort(
      begin(doc_ids),
      Head(doc_ids, 5).end(),
      end(doc_ids),
    [&docid_count](int64_t lhs, int64_t rhs) {
    return pair(docid_count[lhs], -lhs) > pair(docid_count[rhs], -rhs);
    }
    );
    search_results_output << current_query << ':';
    for (auto docid: Head(doc_ids, 5)) 
    {
    const size_t hit_count = docid_count[docid];
     if (hit_count == 0) break;
        search_results_output << " {"
                              << "docid: " << docid << ", "
                              << "hitcount: " << hit_count << '}';
    }
    search_results_output << '\n';
  }
} 
}

void SearchServer::AddQueriesStream(istream& query_input, ostream& search_results_output)
{
  tasks_.push_back(async(Do_Search, ref(query_input), ref(search_results_output), ref(index)));
}