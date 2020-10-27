#include "test_runner.h"

#include <cstddef>  // нужно для nullptr_t
#include <algorithm>
using namespace std;

// Реализуйте шаблон класса UniquePtr
template <typename T>
class UniquePtr {
private:
  T* data_ = nullptr;
public:
  UniquePtr(){}
  UniquePtr(T * ptr) : data_(ptr) {}

  UniquePtr(const UniquePtr&) = delete;
  UniquePtr& operator = (const UniquePtr&) = delete;

  UniquePtr(UniquePtr&& other) : data_(other.Release()) {}

  UniquePtr& operator = (nullptr_t) {
      Reset(nullptr);
      return *this;
    }
  UniquePtr& operator = (UniquePtr&& other) {
    if(data_) delete data_;
    data_ = other.Release();
    return *this;
  }
  ~UniquePtr() { delete data_ ;}

  T& operator * () const { if (data_) return *(data_); }

  T * operator -> () const { return data_; }

  T * Release() {
    T* tmp = data_;
    if (data_ != nullptr)
      data_ = nullptr;
    return tmp;
  }

  void Reset(T * ptr) {
    if (ptr != this -> data_) {
      T* old = this -> data_;
      this -> data_ = ptr;
      delete old;
    } 
  }

  void Swap(UniquePtr& other) { if (this != &other) swap(this -> data_, other.data_); }

  T * Get() const {return data_;}
};


struct Item {
  static int counter;
  int value;
  Item(int v = 0): value(v) {
    ++counter;
  }
  Item(const Item& other): value(other.value) {
    ++counter;
  }
  ~Item() {
    --counter;
  }
};

int Item::counter = 0;


void TestLifetime() {
  Item::counter = 0;
  {
    UniquePtr<Item> ptr(new Item);
    ASSERT_EQUAL(Item::counter, 1);

    ptr.Reset(new Item);
    ASSERT_EQUAL(Item::counter, 1);
  }
  ASSERT_EQUAL(Item::counter, 0);

  {
    UniquePtr<Item> ptr(new Item);
    ASSERT_EQUAL(Item::counter, 1);

    auto rawPtr = ptr.Release();
    ASSERT_EQUAL(Item::counter, 1);

    delete rawPtr;
    ASSERT_EQUAL(Item::counter, 0);
  }
  ASSERT_EQUAL(Item::counter, 0);
}

void TestGetters() {
  UniquePtr<Item> ptr(new Item(42));
  ASSERT_EQUAL(ptr.Get()->value, 42);
  ASSERT_EQUAL((*ptr).value, 42);
  ASSERT_EQUAL(ptr->value, 42);
}

int main() {
  TestRunner tr;
  RUN_TEST(tr, TestLifetime);
  RUN_TEST(tr, TestGetters);
}
