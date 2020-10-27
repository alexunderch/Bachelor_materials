#pragma once
#include <cstdlib>
#include <algorithm>
#include <iostream>

template <typename T>
class SimpleVector {
public:
  SimpleVector() : data(nullptr), end_(nullptr), __size(0), __capacity(0) {};
  explicit SimpleVector(size_t size) { data = new T[size];
                                      end_ = data + size;
                                      __capacity = size; __size = size;}
  SimpleVector(const SimpleVector<T>& other) : data(new T[other.__capacity])
                                             , __size(other.__size)
                                             , end_(data + __size;)
                                             , __capacity(other.__capacity)
  {
    copy(other(begin), other(end), begin());
  }

  void operator = (const SimpleVector<T>& to_assign)
  {
    if (to_assign.__size() <= __capacity)
    std::copy(begin(to_assign), end(to_assign), begin);
    else
    {
        SimpleVector<T> tmp(to_assign);
        std::swap(tmp.data, data);
        std::swap(tmp.__size, __size);
        std::swap(tmp.end_, end_);
        std::swap(tmp.__capacity, __capacity);
    }
  }


  ~SimpleVector() { delete[] data; }

  T& operator[](size_t index) { return data[index]; }

  const T& operator[](size_t index) const { return data[index]; }

  const T* begin() const { return data; }
  const T* end() const  { return end_; }

  T* begin() { return data; }
  T* end()  { return end_; }

  size_t Size() const { return __size; }
  size_t Capacity() const { return __capacity; }
  void PushBack(const T& value)
  {
    if (__capacity == 0)
    {
      data = new T[1];
      end_= data + 1;
      *data = value;
      __capacity = 1;
      __size = 1;
    }
    else

    {
      if (Size() == Capacity())
      {
        __capacity *= 2;
        T* new_data = new T[__capacity];
        std::copy(data, end_, new_data);
        delete[] data;
        data = new_data;
      }
      data[__size] = value;
      __size++;
      end_  = data + __size;
  }
  }

private:
  T* data;
  T* end_;
  size_t __size;
  size_t __capacity;
};
