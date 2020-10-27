#include "test_runner.h"

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <unordered_map>

using namespace std;

struct Record {
  string id;
  string title;
  string user;
  int timestamp;
  int karma;
  bool operator == (const Record& other) const{
    return (id == other.id) && (title == other.title) &&
           (user == other.user) && (timestamp == other.timestamp) &&
           (karma == other.karma);
  }
};
struct Iterator {
   multimap <int, const Record*>::iterator karmas;
   multimap <int, const Record*>::iterator times;
   multimap <string, const Record*>::iterator users;

};
// Реализуйте этот класс
class Database {
public:

  bool Put(const Record& record) {
    const auto paste = data_all.emplace(record.id, record);

    if (!paste.second) return false;

    const auto k = data_karma.emplace(record.karma, &(*paste.first));
    const auto u = data_timestamp.emplace(record.timestamp, move(&data_all[record.id]));
    const auto t = data_user.emplace(record.user, move(&data_all[record.id]));
    iters[record.id] = {k, u, t};
    return true;
  }
  const Record* GetById(const string& id) const {
    
    if (!data_all.count(id)) return nullptr;

    return &(data_all.at(id));
  }

  bool Erase(const string& id) {

    if (!data_all.count(id)) return false;
  
    data_karma.erase(iters[id].karmas);
    data_timestamp.erase(iters[id].times);
    data_user.erase(iters[id].users);

    
    iters.erase(id);
    data_all.erase(id);
    return true;
  }
  
   template <typename Callback>
  void RangeByTimestamp(int low, int high, Callback callback) const {
    const auto begin = data_timestamp.lower_bound(low);
    const auto end = data_timestamp.upper_bound(high);
     for (auto it = begin; it != end; it++){
        if(!callback(*(it -> second))) break;
     }
  }
 
  template <typename Callback>
  void RangeByKarma(int low, int high, Callback callback) const {
    const auto begin = data_karma.lower_bound(low);
    const auto end = data_karma.upper_bound(high);
 
    for (auto it = begin; it != end; it++){
      if(!callback(*(it -> second))) return;
     }
  }
 
  template <typename Callback>
  void AllByUser(string user, Callback callback) const {
    const auto eq = data_user.equal_range(user);
 
    for (auto it = eq.first; it != eq.second; it++){
      if(!callback(*(it -> second))) return;
     }

  }
  
private:
  unordered_map <string, Record> data_all;
  unordered_map <string, Iterator> iters;
  multimap <int, const Record*> data_karma;
  multimap <int, const Record*> data_timestamp;
  multimap <string, const Record*> data_user; 
};

void TestRangeBoundaries() {
  const int good_karma = 1000;
  const int bad_karma = -10;

  Database db;
  db.Put({"id1", "Hello there", "master", 1536107260, good_karma});
  db.Put({"id2", "O>>-<", "general2", 1536107260, bad_karma});

  int count = 0;
  db.RangeByKarma(bad_karma, good_karma, [&count](const Record&) {
    ++count;
    return true;
  });

  ASSERT_EQUAL(2, count);
}

void TestSameUser() {
  Database db;
  db.Put({"id1", "Don't sell", "master", 1536107260, 1000});
  db.Put({"id2", "Rethink life", "master", 1536107260, 2000});

  int count = 0;
  db.AllByUser("master", [&count](const Record&) {
    ++count;
    return true;
  });

  ASSERT_EQUAL(2, count);
}

void TestReplacement() {
  const string final_body = "Feeling sad";

  Database db;
  db.Put({"id", "Have a hand", "not-master", 1536107260, 10});
  db.Erase("id");
  db.Put({"id", final_body, "not-master", 1536107260, -10});

  auto record = db.GetById("id");
  ASSERT(record != nullptr);
  ASSERT_EQUAL(final_body, record->title);
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, TestRangeBoundaries);
  RUN_TEST(tr, TestSameUser);
  RUN_TEST(tr, TestReplacement);
  return 0;
}
