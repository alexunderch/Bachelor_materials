#include "test_runner.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <queue>
#include <stdexcept>
#include <set>
using namespace std;

template <class T>
class ObjectPool {
public:
  T* Allocate()
  {
    if (objects.size()) {
      auto x = objects.front();
      selected.insert(x);
      objects.pop();
      return x;
    }
    else {
      T* new_object = new T;
      selected.insert(new_object);
      return new_object;
    }
  }
  T* TryAllocate()
  {
    if (objects.empty()) {return nullptr;}
    else {
      T* new_object = objects.front();
      selected.insert(new_object);
      objects.pop();
      return new_object;
    }
  }

  void Deallocate(T* object)
  {
    auto res = selected.lower_bound(object);
    if (res == end(selected)) throw invalid_argument("ouch");
    else {
      objects.push(object);
      selected.erase(object);
    }
  }

  ~ObjectPool()
  {

    for(auto it = selected.begin(); it != selected.end(); ++it) {delete *it;}
    selected.erase(begin(selected), end(selected));

    while (!objects.empty()) { delete objects.front(); objects.pop(); }

  }


private:
  queue<T*> objects;
  set<T*> selected;
};

void TestObjectPool() {
  ObjectPool<string> pool;

  auto p1 = pool.Allocate();
  auto p2 = pool.Allocate();
  auto p3 = pool.Allocate();

  *p1 = "first";
  *p2 = "second";
  *p3 = "third";

  pool.Deallocate(p2);
  ASSERT_EQUAL(*pool.Allocate(), "second");

  pool.Deallocate(p3);
  pool.Deallocate(p1);
  ASSERT_EQUAL(*pool.Allocate(), "third");
  ASSERT_EQUAL(*pool.Allocate(), "first");

  pool.Deallocate(p1);
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, TestObjectPool);
  return 0;
}
