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
/*
int LGD (int a, int b) {
  if (b == 0) {
    return a;
  } else {
    return LGD(b, a % b);
  }
}


int Abs(int& a)
{
  if (a < 0) return -a;
  return a;
}

class Rational {
public:
    Rational() {
        // Реализуйте конструктор по умолчанию
        p = 0; //no mistake
        q = 1;
    }
    Rational(int numerator, int denominator) {
        // Реализуйте конструктор
        int sign = 1;

        if (numerator == 0) denominator = 1;
        if (denominator == 0)
        {
          throw invalid_argument("Invalid argument");
        }
        if ((numerator < 0 && denominator > 0) ||
           (numerator > 0 && denominator < 0))
        {
          sign = -1;
        }
        numerator = sign * Abs(numerator);
        denominator = Abs(denominator);

        int lgd = LGD (Abs(numerator), Abs(denominator));
        p = numerator / lgd;
        q = denominator / lgd;
    }
    int Numerator() const {
        // Реализуйте этот метод
        return p;
    }
    int Denominator() const {
        // Реализуйте этот метод
        return q;
    }
private:
    // Добавьте поля
    int p;
    int q;
};


// Реализуйте для класса Rational операторы ==, + и -
bool operator == (const Rational& l, const Rational& r)
{
  if (l.Numerator() == r.Numerator() && l.Denominator() == r.Denominator())
  {
    return true;
  }
  return false;
}
Rational operator + (Rational a, Rational b)
{
    int num  = (a.Numerator() * b.Denominator()) + (b.Numerator() * a.Denominator());
    int denom = a.Denominator() * b.Denominator();
    return Rational(num, denom);
}
Rational operator - (Rational a, Rational b)
{
    int num = (a.Numerator() * b.Denominator()) - (b.Numerator() * a.Denominator());
    int denom = a.Denominator() * b.Denominator();
    return Rational(num, denom);
}
// Реализуйте для класса Rational операторы * и /
Rational operator * (Rational a, Rational b)
{
  return Rational(a.Numerator() * b.Numerator(), a.Denominator() * b. Denominator());
}

Rational operator / (Rational a, Rational b)
{
  if (b.Numerator() == 0)
  {
    throw domain_error("Division by zero");
  }
  return Rational(a.Numerator() * b.Denominator(), a.Denominator() * b. Numerator());
}

// Реализуйте для класса Rational операторы << и >>
istream& operator >> (istream& stream, Rational& r)
{
  int p, q;
  if (stream >> p && stream.ignore(1) && stream >> q)
  {
    r = Rational(p, q);
  }
  return stream;
}

ostream& operator << (ostream& stream, const Rational& r)
{
  stream << r.Numerator() << "/" << r.Denominator();
  return stream;
}

// Реализуйте для класса Rational оператор(ы), необходимые для использования его
// в качестве ключа map'а и элемента set'а

bool operator > (Rational lhs, Rational rhs)
{
  if (lhs.Denominator() == rhs.Denominator())
  {
    return (lhs.Numerator() > rhs.Numerator());
  }
  else
  {
    return (lhs.Numerator() * rhs.Denominator()) > (rhs.Numerator() * lhs.Denominator());
  }
}
bool operator < (Rational lhs, Rational rhs)
{
  if (lhs.Denominator() == rhs.Denominator())
  {
    return (lhs.Numerator() < rhs.Numerator());
  }
  else
  {
    return (lhs.Numerator() * rhs.Denominator()) < (rhs.Numerator() * lhs.Denominator());
  }
}
*/
void DefaultConstuctorTest()
{
  Rational r;
  Rational exp(0, 1);
  AssertEqual(r.Numerator(), exp.Numerator(), "There isn't by default!");
  AssertEqual(r.Denominator(), exp.Denominator(), "There isn't by default!");
}

void AbbrevaiteFractionTest()
{
  AssertEqual(Rational(2, 4).Numerator(), Rational(1,2).Numerator(),"There should be Numerator 1 here!");
  AssertEqual(Rational(2, 4).Denominator(), Rational(1,2).Denominator(),"There should be Denominator 2 here!");

  AssertEqual(Rational(3, 8).Numerator(), Rational(3,8).Numerator(),"There should be Numerator 3 here!");
  AssertEqual(Rational(3, 8).Denominator(), Rational(3,8).Denominator(),"There should be Denominator 8 here!");

  AssertEqual(Rational(-3, 112).Numerator(), Rational(-3,112).Numerator(),"There should be Numerator -3 here!");
  AssertEqual(Rational(-3, 112).Numerator(), Rational(-3,112).Numerator(),"There should be Denominator 112 here!");

  AssertEqual(Rational(3, -8).Denominator(), Rational(-3,8).Denominator(),"There should be Numerator 8 here!");
  AssertEqual(Rational(3, -8).Denominator(), Rational(-3,8).Denominator(),"There should be Denominator 8 here!");

  AssertEqual(Rational(-6, -7).Denominator(), Rational(6,7).Denominator(),"There should be Numerator 6 here!");
  AssertEqual(Rational(-6, -7).Denominator(), Rational(6, 7).Denominator(),"There should be denominator 7 here!");

  AssertEqual(Rational(0, 100).Numerator(), 0, "Canonical form of 0/100 is 0/1");
  AssertEqual(Rational(0, 100).Denominator(), 1, "Canonical form of 0/100 is 0/1");

  AssertEqual(Rational(2147483647, 2147483647).Numerator(), 1, "Canonical form of 2147483647/2147483647 is 1/1");
  AssertEqual(Rational(2147483647, 2147483647).Denominator(), 1, "Canonical form of 2147483647/2147483647 is 1/1");
}


int main() {
  TestRunner runner;
  runner.RunTest(DefaultConstuctorTest, "default constr");
  runner.RunTest(AbbrevaiteFractionTest, "AbbrevaiteFractionTest");
  // добавьте сюда свои тесты
  return 0;
}
