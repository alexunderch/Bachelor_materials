#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <cmath>
#include <iomanip>

using namespace std;

class Figure
{
public:
  virtual string Name() const = 0;
  virtual double  Perimeter () const = 0;
  virtual double Area() const = 0;
};

class Triangle : public Figure
{
public:
  Triangle(const double& a, const double& b, const double& c) : _a(a), _b(b), _c(c) {}

  string Name() const override
  {
    return "TRIANGLE";
  }

  double  Perimeter () const override
  {
    return _a + _b + _c;
  }

  double Area() const override
  {
    double p = (_a + _b + _c) / 2;
    return sqrt(p * (p - _a) * (p - _b) * (p - _c));
  }
private:
  const double _a, _b, _c;
};

class Rect : public Figure
{
public:
  Rect(const double& weight, const double& height) : _weight(weight), _height(height) {}

  string Name() const override
  {
    return "RECT";
  }

  double  Perimeter () const override
  {
    return 2 * (_weight + _height);
  }

  double Area() const override
  {
    return _weight * _height;
  }

private:
  double _weight, _height;
};

class Circle : public Figure
{
public:
  Circle (const double& rad) : _r(rad) {}

  string Name() const override
  {
    return "CIRCLE";
  }

  double  Perimeter () const override
  {
    return 2 * 3.14 * _r;
  }

  double Area() const override
  {
    return 3.14 * _r * _r;
  }

private:
  const double _r;
};

shared_ptr<Figure> CreateFigure(istringstream& is)
{
  string op;
  is >> op;

  if (op == "TRIANGLE")
  {
    double a = 0, b = 0, c = 0;
    is >> a >> b >> c;
    return make_shared<Triangle>(a, b, c);
  }
  else if (op == "RECT")
  {
    double weight, height;
    is >> weight >> height;
    return make_shared<Rect>(weight, height);
  }
  else if (op == "CIRCLE")
  {
    double rad;
    is >> rad;
    return make_shared<Circle>(rad);
  }
  return NULL;
}



int main() {
  vector<shared_ptr<Figure>> figures;
  for (string line; getline(cin, line); ) {
    istringstream is(line);

    string command;
    is >> command;
    if (command == "ADD") {
      figures.push_back(CreateFigure(is));
    } else if (command == "PRINT") {
      for (const auto& current_figure : figures) {
        cout << fixed << setprecision(3)
             << current_figure->Name() << " "
             << current_figure->Perimeter() << " "
             << current_figure->Area() << endl;
      }
    }
  }
  return 0;
}
