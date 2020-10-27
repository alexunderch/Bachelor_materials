#include "test_runner.h"
#include "profile.h"

#include <map>
#include <string>
#include <functional>
#include <utility>
#include <string_view>
#include <iostream>
#include <future>
#include <vector>
#include <sstream>
#include <set>

using namespace std;

template <typename Iterator>
class IteratorRange {
public:
  IteratorRange(Iterator begin, Iterator end)
    : first(begin)
    , last(end)
    , size_(distance(first, last))
  {
  }

  Iterator begin() const {
    return first;
  }

  Iterator end() const {
    return last;
  }

  size_t size() const {
    return size_;
  }

private:
  Iterator first, last;
  size_t size_;
};

template <typename Iterator>
class Paginator {
private:
  vector<IteratorRange<Iterator>> pages;

public:
  Paginator(Iterator begin, Iterator end, size_t page_size) {
    for (size_t left = distance(begin, end); left > 0; ) {
      size_t current_page_size = min(page_size, left);
      Iterator current_page_end = next(begin, current_page_size);
      pages.push_back({begin, current_page_end});

      left -= current_page_size;
      begin = current_page_end;
    }
  }

  auto begin() const {
    return pages.begin();
  }

  auto end() const {
    return pages.end();
  }

  size_t size() const {
    return pages.size();
  }
};

template <typename C>
auto Paginate(C& c, size_t page_size) {
  return Paginator(begin(c), end(c), page_size);
}


struct Stats {
  map<string, int> word_frequences;

  void operator += (const Stats& other)
  {
    for (const auto& [word, frequency]: other.word_frequences)
        word_frequences[word] += frequency;
  }
};

Stats ExploreLine(const set<string>& key_words, const string& line) {
  string_view s = line;
    Stats stats = {};
    size_t start = s.find_first_not_of(" \f\n\r\t\v");
    s.remove_prefix(start);

    while(true) {
        size_t space = s.find(' ');
        string word(s.substr(0, space));
        if (key_words.count(word) > 0) {
            ++stats.word_frequences[word];
        }
        if (space == s.npos) {
            break;
        } else {
            s.remove_prefix(space + 1);
        }
    }
    return stats;
}

  //result += ExploreLine(key_words, line);

template <typename ContainerOfVectors>
Stats ExploreKeyWordsSingleThread(const set<string>& key_words, ContainerOfVectors& input)
{
  Stats result;
  for (auto& word: input) {
    if (key_words.count(word) > 0) { result.word_frequences[word]++; }
}
  return result;
}

Stats ExploreKeyWords(const set<string>& key_words, istream& input)
{
  // Реализуйте эту функцию
  vector<string> v;
  for (string line; input >> line;) { v.push_back(move(line)); }

  vector<future<Stats>> futures;
  for (auto page : Paginate(v, 1))
  {
    futures.push_back(async([&key_words, page] {return ExploreKeyWordsSingleThread(key_words, page);}));
  }
  Stats result = {};
  for (auto& f : futures) { result += f.get(); }
  return result;
}

void TestBasic() {
  const set<string> key_words = {"yangle", "rocks", "sucks", "all"};

  stringstream ss;
  ss << "this new yangle service really rocks\n";
  ss << "It sucks when yangle isn't available\n";
  ss << "10 reasons why yangle is the best IT company\n";
  ss << "yangle rocks others suck\n";
  ss << "Goondex really sucks, but yangle rocks. Use yangle\n";

  const auto stats = ExploreKeyWords(key_words, ss);
  const map<string, int> expected = {
    {"yangle", 6},
    {"rocks", 2},
    {"sucks", 1}
  };
  ASSERT_EQUAL(stats.word_frequences, expected);
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, TestBasic);
}
