#include <cstdint>
#include <utility>
#include <algorithm>
using namespace std;

// Реализуйте SimpleVector в этом файле
// и отправьте его на проверку

template <typename T>
class SimpleVector {
public:
  SimpleVector() = default;
  explicit SimpleVector(size_t vsize);
  SimpleVector(const SimpleVector<T>& other) = delete; //copy ctor;
  //void operator = (const SimpleVector<T>& other);
  SimpleVector(SimpleVector <T>&& other); // move ctor;
  void operator  = (SimpleVector<T>&& other);
  ~SimpleVector();

  T& operator[] (size_t index);
  const T& operator[] (size_t index) const;

  T* begin();
  T* end();
  const T* begin() const;
  const T* end () const;

  size_t Size() const;
  size_t Capacity() const;
  void PushBack(T value);

private:
  T* data = nullptr;
  size_t size = 0;
  size_t capacity = 0;
};

template <typename T>
SimpleVector<T>::SimpleVector(size_t vsize) : data(new T[vsize])
                                            , size(vsize)
                                            , capacity(vsize)
                                             {}
/*
template <typename T>
SimpleVector<T>::SimpleVector(const SimpleVector<T>& other)
                                                  : data(new T[other.capacity])
                                                  , size(other.size)
                                                  , capacity(other.capacity)

{ std::copy(other.begin(), other.end(), begin());}

template <typename T>
void SimpleVector<T>::operator = (const SimpleVector<T>& other)
{
  if (other.size <= capacity) {
    std::copy(other.begin(), other.end(), begin());
    size = other.size;
  } else {
    SimpleVector<T> tmp(other);
    std::swap(tmp.data, data);
    std::swap(tmp.size, size);
    std::swap(tmp.capacity, capacity);
  }
}
*/
template<typename T>
SimpleVector<T>::SimpleVector(SimpleVector <T>&& other)
                                                  : data(new T[other.capacity])
                                                  , size(other.size)
                                                  , capacity(other.capacity)
{
  other.data = nullptr;
  other.size = other.capacity = 0;
}
template <typename T>
void SimpleVector<T>::operator  = (SimpleVector<T>&& other)
{
  delete[] data;
  data = other.data;
  size = other.size;
  capacity = other.capacity;

  other.data = nullptr;
  other.size = other.capacity = 0;
}

template <typename T>
SimpleVector<T>::~SimpleVector() {delete[] data;}

template <typename T>
T& SimpleVector<T>::operator[] (size_t index) {return data[index]; }

template <typename T>
const T& SimpleVector<T>::operator[] (size_t index) const {return data[index];}

template <typename T>
T* SimpleVector<T>::begin() {return data;}

template <typename T>
const T* SimpleVector<T>::begin() const {return data;}

template <typename T>
T* SimpleVector<T>::end() {return data + size;}

template <typename T>
const T* SimpleVector<T>::end() const {return data + size;}

template <typename T>
size_t SimpleVector<T>::Size() const {return size;}

template <typename T>
size_t SimpleVector<T>::Capacity() const {return capacity;}

template <typename T>
void SimpleVector<T>::PushBack(T value)
{
  if (size >= capacity) {
    size_t new_capacity = capacity == 0 ? 1 : 2 * capacity;
    T* new_data = new T[new_capacity];
    std::move(begin(), end(), new_data);
    delete[] data;
    data = new_data;
    capacity = new_capacity;
  }
  data[size++] = std::move(value);
}
