//система ведения личного бюджета
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

#define Macros(Name)         					\
	struct Name {								\
		int value;								\
		explicit Name(int v) : value(v) {}		\
		operator int() const { return value; }	\
	};

    
Macros (Year);
Macros (Month);
Macros (Day);



struct Date {
public:
    Date() {}
    Date (Year y, Month m, Day d): _year(y), _month(m), _day(d) {}
    

    time_t AsTimestamp() const {
        std::tm t;
        t.tm_sec   = 0;
        t.tm_min   = 0;
        t.tm_hour  = 0;
        t.tm_mday  = _day;
        t.tm_mon   = _month- 1;
        t.tm_year  = _year - 1900;
        t.tm_isdst = 0;
    return mktime(&t);
    }

    friend istream& operator >> (istream& is, Date& date);
private:
    Year _year = Year{0};
    Month _month = Month{1};
    Day _day = Day{1};
};


istream& operator>>(istream& is, Date& date) {
	is >> date._year.value;
    is.ignore();
	is >> date._month.value;
	is.ignore();
	is >> date._day.value;
	return is;
}

int ComputeDaysDiff(const Date& date_to, const Date& date_from) {
	const time_t timestamp_to = date_to.AsTimestamp();
	const time_t timestamp_from = date_from.AsTimestamp();
	static const int SECONDS_IN_DAY = 60 * 60 * 24;
	return (timestamp_to - timestamp_from) / SECONDS_IN_DAY;
}


int Days_since2000(const Date& date) {
	static const time_t timestamp_from = Date{ Year{ 2000 }, Month{ 1 }, Day{ 1 } }.AsTimestamp();
	static const int SECONDS_IN_DAY = 60 * 60 * 24;
	const time_t timestamp_to = date.AsTimestamp();
	return (timestamp_to - timestamp_from) / SECONDS_IN_DAY;
}

const int MAX_D = 365 * 100; 


class BudgetManager {
public:
    BudgetManager () : _budget(MAX_D) {}

    void Earn(const Date& date_from, const Date& date_to, double value) {
        double income_per_day = value / (ComputeDaysDiff(date_to, date_from) + 1);
        int start = Days_since2000(date_from), end = Days_since2000(date_to);
        for (int i = start; i <= end; ++i) {
            _budget[i].earned += income_per_day;
        }
    }

    void Spend(const Date& date_from, const Date& date_to, double value) {
        double outcome_per_day = value / (ComputeDaysDiff(date_to, date_from) + 1);
        int start = Days_since2000(date_from), end = Days_since2000(date_to);
        for (int i = start; i <= end; ++i) {
            _budget[i].spent += outcome_per_day;
        }
    }

    void PayTax (const Date& date_from, const Date& date_to, double percentage) {
        double tax = (100 - percentage) / 100.0;
        int start = Days_since2000(date_from), end = Days_since2000(date_to);
        for (int i = start; i <= end; ++i) {
            _budget[i].earned *= tax;
        }
    }
    double ComputeIncome (const Date& date_from, const Date& date_to) {
        double sum = 0  ;
        int start = Days_since2000(date_from), end = Days_since2000(date_to);
        for (int i = start; i <= end; ++i) {
            sum += _budget[i].ComputeDiff();
        }
        return sum;
    }
private:
    struct Stats {
        double earned;
        double spent;
        double ComputeDiff() const { return earned - spent; }
    };
    
    vector<Stats> _budget;
};

int main() {
    int n = 0;
    cin >> n;
    BudgetManager bm;
    for (int i = 0; i < n; ++i){
        string query;
        Date f, t;
        cin >> query >> f >> t;
        if (query == "Earn") {
            double value;
            cin >> value;
            bm.Earn(f, t, value);
        } else if (query == "PayTax") {
            double percentage;
            cin >> percentage;
            bm.PayTax(f, t, percentage);
        } else if (query == "ComputeIncome") {
            cout << fixed << setprecision(25) << bm.ComputeIncome(f, t) << '\n';
        } else if (query == "Spend") {
            double value;
			cin >> value;
			bm.Spend(f, t, value);
        } else {
            cout << "Wrong query!" << endl;
        }
    }
    return 0;
}
