#pragma once
#include "synchronized.h"
#include <future>
#include <istream>
#include <ostream>
#include <set>
#include <deque>
#include <vector>
#include <map>
#include <string>
using namespace std;

class InvertedIndex {
public:
  struct Stats
  {
    size_t doc_id, hitcount;
  };
  InvertedIndex() = default;
  explicit InvertedIndex (istream& document_input);
  const vector<Stats>& Lookup(string_view word) const;

  const deque<string>& GetDocuments() const {
    return docs;
  }

private:
  map<string_view, vector<Stats>> index;
  deque<string> docs;
};

class SearchServer {
public:
  SearchServer() = default;
  explicit SearchServer(istream& document_input) : index(InvertedIndex(document_input)) {}
  void UpdateDocumentBase(istream& document_input);
  void AddQueriesStream(istream& query_input, ostream& search_results_output);

private:
  Synchronized<InvertedIndex> index;
  vector<future<void>> tasks_;
};
