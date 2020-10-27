#include "Common.h"
#include <string>
#include <list>
#include <unordered_map>
#include <mutex>
#include <iostream>
using namespace std;

class LruCache : public ICache {
public:
  LruCache(
      shared_ptr<IBooksUnpacker> books_unpacker,
      const Settings& settings
  ): books_unpacker_(move(books_unpacker)), settings_(move(settings)) {}

  BookPtr GetBook(const string& book_name) override {
    
    lock_guard<mutex> guard(m_);
    
    if (auto it = books_.find(book_name); it != books_.end()) {
      cache_.splice(cache_.begin(), cache_, it -> second);
      return cache_.front();
    }


    auto BOOK = books_unpacker_ -> UnpackBook(book_name);


    if (BOOK -> GetContent().size() > settings_.max_memory) {
      cache_.clear();
      books_.clear();
      sum_rate = 0;
      return BOOK;
    }

    sum_rate += BOOK -> GetContent().size();
   
    
    if (sum_rate > settings_.max_memory) {
      while (!cache_.empty() && sum_rate > settings_.max_memory) {
        sum_rate -= cache_.back().get() -> GetContent().size();
        string name = cache_.back().get() -> GetName();
        cache_.pop_back();
        books_.erase(name);
      }   
    }
    
    cache_.push_front(move(BOOK));
    books_[book_name] = cache_.begin();
    return cache_.front();  
    
  }
private:

  size_t sum_rate = 0;
  const Settings settings_;
  shared_ptr<IBooksUnpacker> books_unpacker_;
  list<BookPtr> cache_;
  unordered_map<string, list<BookPtr>::iterator> books_;
  mutable mutex m_;
};


unique_ptr<ICache> MakeCache(
    shared_ptr<IBooksUnpacker> books_unpacker,
    const ICache::Settings& settings
) {
  mutex m_;
  lock_guard<mutex> guard(m_);
  return make_unique<LruCache>(books_unpacker, settings);
}
/*
#include "Common.h"

#include <algorithm>
#include <cassert>
#include <future>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>


using namespace std;

class LruCache : public ICache {
public:
  LruCache(
      std::shared_ptr<IBooksUnpacker> books_unpacker,
      const Settings& settings
  )
    : books_unpacker_(move(books_unpacker))
    , settings_(settings)
  {}

  BookPtr GetBook(const string& book_name) override {
    lock_guard guard(mutex_);

    auto it = books_.find(book_name);

    if (it == books_.end()) {
      Entry entry;
      entry.book = books_unpacker_->UnpackBook(book_name);

      auto book_size = entry.book->GetContent().size();

      while (!books_.empty() && cache_size_ + book_size > settings_.max_memory) {
        RemoveLruEntry();
      }
      assert(!books_.empty() || cache_size_ == 0);

      if (book_size > settings_.max_memory) {
        return move(entry.book);
      }

      it = AddEntry(book_name, move(entry));
    }

    it->second.rank = current_rank_++;

    return it->second.book;
  }

private:
  struct Entry {
    BookPtr book;
    int rank = 0;
  };

  using Entries = unordered_map<string, Entry>;

private:
  void RemoveLruEntry() {
    assert(!books_.empty());

    auto it = min_element(
        books_.begin(), books_.end(),
        [](const Entries::value_type& lhs, const Entries::value_type& rhs) {
          return lhs.second.rank < rhs.second.rank;
        }
    );

    cache_size_ -= it->second.book->GetContent().size();
    books_.erase(it);
  }

  Entries::iterator AddEntry(const string& book_name, Entry entry) {
    assert(0 == books_.count(book_name));
    cache_size_ += entry.book->GetContent().size();
    return books_.insert({book_name, move(entry)}).first;
  }

private:
  std::shared_ptr<IBooksUnpacker> books_unpacker_;
  const Settings settings_;

  mutex mutex_;
  Entries books_;
  int current_rank_ = 0;
  size_t cache_size_ = 0;
};


unique_ptr<ICache> MakeCache(
    std::shared_ptr<IBooksUnpacker> books_unpacker,
    const ICache::Settings& settings
) {
  return make_unique<LruCache>(move(books_unpacker), settings);
}


*/