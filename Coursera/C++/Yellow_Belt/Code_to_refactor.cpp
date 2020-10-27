#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Person
{
public:
  Person(const string& name, const string& job) : _name(name) , _job(job)
  {}

  string GetName() const
  {
    return _name;
  }

  string GetJob() const
  {
    return _job;
  }

  virtual void Walk(const string& destination) const
  {
    Log() << " walks to: " << destination << endl;
  }
protected:
  virtual ostream& Log() const {
    return cout << GetJob() << ": " << GetName();
  }
private:
  const string _name, _job;
};

class Student : public Person
{
public:

    Student(const string& name, const string& favouriteSong) : Person(name, "Student"),
                                                               FavouriteSong(favouriteSong)
    {}

    void Learn() const
    {
        cout << "Student: " << GetName() << " learns" << endl;
    }

    void Walk(const string& destination) const override
    {
        Person :: Walk(destination);
        SingSong();
    }

    void SingSong() const
    {
        Log() << " sings a song: " << FavouriteSong << endl;
    }

private:
    const string FavouriteSong;
};


class Teacher : public Person
{
public:
    Teacher(const string& name, const string& subject) : Person(name, "Teacher"),
                                                          Subject(subject)
    {}

    void Teach() const
    {
        Log() << " teaches: " << Subject << endl;
    }

private:
    const string Subject;
};


class Policeman : public Person
{
public:
    Policeman(const string& name) : Person(name, "Policeman")
    {}

    void Check(const Person& p) const
    {
        cout << "Policeman: " << GetName() << " checks " << p.GetJob() << "."
        << p.GetJob() << "'s name is: " << p.GetName() << endl;
    }
};


void VisitPlaces(Person& person, const vector<string>& places) {
    for (const auto& p : places) {
        person.Walk(p);
    }
}


int main() {
    Teacher t("Jim", "Math");
    Student s("Ann", "We will rock you");
    Policeman p("Bob");

    VisitPlaces(t, {"Moscow", "London"});
    p.Check(s);
    VisitPlaces(s, {"Moscow", "London"});
    return 0;
}
