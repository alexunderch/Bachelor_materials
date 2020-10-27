#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <ctime>

using namespace std;

template <class T>
ostream& operator << (ostream& os, const vector<T>& s) {
  os << "{";
  bool first = true;
  for (const auto& x : s) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << x;
  }
  return os << "}";
}

template <class T>
ostream& operator << (ostream& os, const set<T>& s) {
  os << "{";
  bool first = true;
  for (const auto& x : s) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << x;
  }
  return os << "}";
}

template <class K, class V>
ostream& operator << (ostream& os, const map<K, V>& m) {
  os << "{";
  bool first = true;
  for (const auto& kv : m) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << kv.first << ": " << kv.second;
  }
  return os << "}";
}

template<class T, class U>
void AssertEqual(const T& t, const U& u, const string& hint = {}) {
  if (t != u) {
    ostringstream os;
    os << "Assertion failed: " << t << " != " << u;
    if (!hint.empty()) {
       os << " hint: " << hint;
    }
    throw runtime_error(os.str());
  }
}

void Assert(bool b, const string& hint) {
  AssertEqual(b, true, hint);
}

class TestRunner {
public:
  template <class TestFunc>
  void RunTest(TestFunc func, const string& test_name) {
    try {
      func();
      cerr << test_name << " OK" << endl;
    } catch (exception& e) {
      ++fail_count;
      cerr << test_name << " fail: " << e.what() << endl;
    } catch (...) {
      ++fail_count;
      cerr << "Unknown exception caught" << endl;
    }
  }

  ~TestRunner() {
    if (fail_count > 0) {
      cerr << fail_count << " unit tests failed. Terminate" << endl;
      exit(1);
    }
  }

private:
  int fail_count = 0;
};

int GetDistinctRealRootCount(double a, double b, double c) {
  // Вы можете вставлять сюда различные реализации функции,
  // чтобы проверить, что ваши тесты пропускают корректный код
  // и ловят некорректный
}


void QuadricEquationTest()
{
  mt19937 gen1;
  uniform_real_distribution<> unif(-100, 100);
  for (int i = 1; i < 100; ++i)
  {
    const auto a = unif(gen1);
    const auto b = unif(gen1);
    const auto c = unif(gen1);
    const auto count = GetDistinctRealRootCount(a, b, c);
    Assert(count >= 0 &&  count <= 2, "The eq should have 0..2 roots");
  }

}

void ZeroRootsTest()
{
  AssertEqual(GetDistinctRealRootCount(1,2, 3), 0,
  "Eq x^2 +2x + 3 = 0 has negative D");
  AssertEqual(GetDistinctRealRootCount(1, 1, 2), 0,
  "Eq x^2 + x + 2 = 0 has negative D");
  AssertEqual(GetDistinctRealRootCount(1, -1, 2), 0,
  "Eq x^2 - x + 2 = 0 has negative D");
  AssertEqual(GetDistinctRealRootCount(-1, -1, -2), 0,
  "Eq -x^2 - x - 2 = 0 has negative D");
}

void OneRootTest()
{
  mt19937 gen1;
  uniform_real_distribution<> unif(-100, 100);
  for (int i = 1; i < 100; ++i)
  {
    const auto x_1 = unif(gen1);

    const auto p = x_1 + x_1;
    const auto q = x_1 * x_1;
    const auto count = GetDistinctRealRootCount(1, p, q);
    stringstream descr;
    descr << "Eq: x^2 " << p << " " << q << " = 0 should have had 1 real root"
    << " and has: " << count;
    AssertEqual(count, 1, descr.str());
    }
}

void LinearEquationTest()
{
  AssertEqual(GetDistinctRealRootCount(0, 1, 2), 1,
  "Eq x + 2 = 0 root is x = -2");
  AssertEqual(GetDistinctRealRootCount(0, -1, 2), 1,
  "Eq -x + 2 = 0 root is x = 2");
  AssertEqual(GetDistinctRealRootCount(0, 1, -2), 1,
  "Eq x - 2 = 0 root is x = 2");
}

void ConstantEquationTest()
{
  AssertEqual(GetDistinctRealRootCount(0, 0 , 2), 0,
  "Eq with a,b = 0, c = 2 has NO REAL ROOTS");
  AssertEqual(GetDistinctRealRootCount(0, 0 , -2), 0,
  "Eq with a,b = 0, c = -2 has NO REAL ROOTS");
  AssertEqual(GetDistinctRealRootCount(0, 0, 123), 0,
  "Eq with a,b = 0, c = 123 has NO REAL ROOTS");
}

int main() {
  TestRunner runner;
  ConstantEquationTest();
  LinearEquationTest();
  OneRootTest();
  ZeroRootsTest();
  QuadricEquationTest();
  // добавьте сюда свои тесты
  return 0;
}
