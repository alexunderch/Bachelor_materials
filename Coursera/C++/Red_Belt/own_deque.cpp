#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
using namespace std;

#define sample()                                 \
if (index - Fronts.size() < Backs.size())        \
{                                                \
  return Backs[index - Fronts.size()];           \
}                                                \
if (index < Fronts.size())                       \
{                                                \
  return Fronts[Fronts.size() - 1 - index];      \
}

template <typename T>
class Deque
{
private:
  vector<T> Fronts;
  vector<T> Backs;
public:
  Deque()
  {
    Fronts.resize(0);
    Backs.resize(0);
  }

  bool Empty () const
  {
    return (Fronts.empty() && Backs.empty());
  }

  size_t Size () const
  {
    return Fronts.size() + Backs.size();
  }

  T& operator [] (size_t index)
  {
    sample();
  }

  T& operator [] (size_t index) const
  {
    sample();
  }

  T& At(size_t index)
  {
    sample();
    throw out_of_range("Detected an effort to access out of range element");
  }

  T& At(size_t index) const
  {
    sample();
    throw out_of_range("Detected an effort to access out of range element");
  }

  T& Front() {
		if (!Fronts.empty()) {
			return Fronts.back();
		}
		else return *(Backs.begin());
	}
	const T& Front() const {

		if (!Fronts.empty()) {
			return Fronts.back();
		}
		else return *(Backs.begin());
	}
	T& Back() {
		if (!Backs.empty()) {
			return Backs.back();
		}
		else return *(Fronts.begin());
	}
	const T& Back() const {
		if (!Backs.empty()) {
			return Backs.back();
		}
		else return *(Fronts.begin());

	}
  void PushFront(T t)
  {
    Fronts.push_back(t);
  //  reverse(Fronts.begin(), Fronts.end());
  }
  void PushBack(T t)
  {
    Backs.push_back(t);
  }

};

int main()
{
  Deque<int> dq_of_int;
  dq_of_int.PushFront(1);
  dq_of_int.PushFront(2);
  dq_of_int.PushFront(3);
  dq_of_int.PushBack(4);
  dq_of_int.PushBack(5);
  return 0;
}
