#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

bool IsPalindrom(const string& s) {
  //
}

void EarlyStringTest()
{
  AssertEqual(IsPalindrom(""), 1, "An empty string is palindrom");
  AssertEqual(IsPalindrom("a"), 1, "An One-char string is palindrom");
  AssertEqual(IsPalindrom("acca"), 1, "acca is palindrom");
  AssertEqual(IsPalindrom(" acca "), 1, "_ acca _ is palindrom");
  AssertEqual(IsPalindrom("aCca"), 0, "aCca isn't palindrom => C != c");
  AssertEqual(IsPalindrom("ab u ba"), 1, "ab u ba is palindrom");
  AssertEqual(IsPalindrom("ab aba u ba"), 0, "ab aba u ba isn't palindrom");
  AssertEqual(IsPalindrom("abbac"), 0, "abbac isn't palindrom");
  AssertEqual(IsPalindrom("cabba"), 0, "cabba isn't palindrom");
  AssertEqual(IsPalindrom("a bba"), 0, "a bba isn't palindrom");
  AssertEqual(IsPalindrom("  aba"), 0, "  aba isn't palindrom");
  AssertEqual(IsPalindrom("aba  "), 0, "aba   isn't palindrom");
  AssertEqual(IsPalindrom("abXYa"), 0, "abXYa isn't palindrom");
  AssertEqual(IsPalindrom("aYbXYa"), 0, "aYbXYa isn't palindrom");
  Assert(!IsPalindrom("XabbaY"), "XabbaY is not a palindrome");
  Assert(IsPalindrom("a b X b a"), "`a b X b a` is a palindrome");


}



int main() {
  TestRunner runner;
  runner.RunTest(EarlyStringTest, " ");
  return 0;
}
