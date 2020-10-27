#include <iostream>
#include <tuple>
#include <string>
#include <map>

using namespace std;
/*
enum class TaskStatus
{
NEW,          // новая
IN_PROGRESS,  // в разработке
TESTING,      // на тестировании
DONE          // завершена
};

// Объявляем тип-синоним для map<TaskStatus, int>,
// позволяющего хранить количество задач каждого статуса
using TasksInfo = map<TaskStatus, int>;
*/
class TeamTasks {
public:
    // Получить статистику по статусам задач конкретного разработчика
    const TasksInfo& GetPersonTasksInfo(const string& person) const
    {
        return tasks_for_person.at(person);
    }

    // Добавить новую задачу (в статусе NEW) для конкретного разработчитка
    void AddNewTask(const string& person)
    {
        ++tasks_for_person[person][TaskStatus::NEW];
    }

    // Обновить статусы по данному количеству задач конкретного разработчика,
    // подробности см. ниже
    tuple<TasksInfo, TasksInfo> PerformPersonTasks(
        const string& person, int task_count)
    {
        // создаём 2 словаря: обновлённых и нетронутых задач
        TasksInfo untouched;
        TasksInfo updated;

        if (tasks_for_person.count(person))
        {
            untouched = tasks_for_person[person];
        }
        else return tie(updated, untouched);

        // статуса DONE быть не толжно - тупo
        untouched.erase(TaskStatus::DONE);

        for (size_t i = 0; (i < 3) && (task_count > 0); ++i)
        {
            // созадим текущий и последующий статусы
            TaskStatus status = static_cast<TaskStatus>(i);
            TaskStatus next_status = static_cast<TaskStatus>(i + 1);

            // условие: если такой статус нетронут и список заданий затрагивает
            // это событие в очереди
            if (untouched.count(status) && task_count >= untouched[status])
            {
                //передвигаем событие в очереди
                updated[next_status] = untouched[status];
                // уменьшаем число событий
                task_count -= untouched[status];
                //раз событие затронуто, то удалим его из такой очереди
                untouched.erase(status);
            }
            else if (untouched.count(status) && task_count < untouched[status])
            {
                updated[next_status] = task_count;
                untouched[status] -= task_count;
                task_count = 0;
            }
            tasks_for_person[person][TaskStatus::NEW] = untouched[TaskStatus::NEW];
            tasks_for_person[person][TaskStatus::IN_PROGRESS] =
            untouched[TaskStatus::IN_PROGRESS] + updated[TaskStatus::IN_PROGRESS];
            tasks_for_person[person][TaskStatus::TESTING] =
            untouched[TaskStatus::TESTING] + updated[TaskStatus::TESTING];
            tasks_for_person[person][TaskStatus::DONE] +=
            updated[TaskStatus::DONE];

        }
        for (size_t i = 0; i <= 3; ++i)
        {
            // созадим текущий и последующий статусы
            TaskStatus status = static_cast<TaskStatus>(i);
            if (tasks_for_person[person][status] == 0) {
                tasks_for_person[person].erase(status);
            }
            if ((i != 3) && !untouched[status]) {
                untouched.erase(status);
            }
            if (i != 0 && !updated[status]) {
                updated.erase(status);
            }
        }
        return tie(updated, untouched);
    }

private:
    map<string, TasksInfo> tasks_for_person;
};

void PrintTasksInfo(TasksInfo tasks_info) {
    cout << tasks_info[TaskStatus::NEW] << " new tasks" <<
        ", " << tasks_info[TaskStatus::IN_PROGRESS] << " tasks in progress" <<
        ", " << tasks_info[TaskStatus::TESTING] << " tasks are being tested" <<
        ", " << tasks_info[TaskStatus::DONE] << " tasks are done" << endl;
}
/*
int main() {
    TeamTasks tasks;
    tasks.AddNewTask("Ilia");
    for (int i = 0; i < 5; ++i) {
        tasks.AddNewTask("Ivan");
    }
    cout << "Ilia's tasks: ";
    PrintTasksInfo(tasks.GetPersonTasksInfo("Ilia"));
    cout << "Ivan's tasks: ";
    PrintTasksInfo(tasks.GetPersonTasksInfo("Ivan"));

    TasksInfo updated_tasks, untouched_tasks;

    tie(updated_tasks, untouched_tasks) =
        tasks.PerformPersonTasks("Ivan", 5);
    cout << "Updated Ivan's tasks: ";
    PrintTasksInfo(updated_tasks);
    cout << "Untouched Ivan's tasks: ";
    PrintTasksInfo(untouched_tasks);

    tie(updated_tasks, untouched_tasks) =
        tasks.PerformPersonTasks("Ivan", 5);
    cout << "Updated Ivan's tasks: ";
    PrintTasksInfo(updated_tasks);
    cout << "Untouched Ivan's tasks: ";
    PrintTasksInfo(untouched_tasks);

    tie(updated_tasks, untouched_tasks) =
        tasks.PerformPersonTasks("Ivan", 1);
    cout << "Updated Ivan's tasks: ";
    PrintTasksInfo(updated_tasks);
    cout << "Untouched Ivan's tasks: ";
    PrintTasksInfo(untouched_tasks);

    for (int i = 0; i < 5; ++i) {
        tasks.AddNewTask("Ivan");
    }
    cout << "Ilia's tasks: ";
    PrintTasksInfo(tasks.GetPersonTasksInfo("Ilia"));
    cout << "Ivan's tasks: ";
    PrintTasksInfo(tasks.GetPersonTasksInfo("Ivan"));

    tie(updated_tasks, untouched_tasks) =
        tasks.PerformPersonTasks("Ivan", 2);
    cout << "Updated Ivan's tasks: ";
    PrintTasksInfo(updated_tasks);
    cout << "Untouched Ivan's tasks: ";
    PrintTasksInfo(untouched_tasks);

    cout << "Ivan's tasks: ";
    PrintTasksInfo(tasks.GetPersonTasksInfo("Ivan"));

    tie(updated_tasks, untouched_tasks) =
        tasks.PerformPersonTasks("Ivan", 4);
    cout << "Updated Ivan's tasks: ";
    PrintTasksInfo(updated_tasks);
    cout << "Untouched Ivan's tasks: ";
    PrintTasksInfo(untouched_tasks);

    cout << "Ivan's tasks: ";
    PrintTasksInfo(tasks.GetPersonTasksInfo("Ivan"));

    return 0;
}*/
