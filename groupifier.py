from dataclasses import dataclass, field
from random import choice, randrange, sample, shuffle, seed
import math
import itertools

INTERESTS = ["sports", "going out", "drinking", "programming", "cooking", "building stuff", "gaming", "board games", "music", "culture", "raising kids", ]
MAX_GROUP_SIZE = 6
POOL_COUNT = 10
POOL_SIZE = 50
STALL_GENERATIONS = 50


person_count = itertools.count()

@dataclass()
class Person:
    name: str
    english: bool
    female: bool
    current_class: str|None
    interests: list[str]
    id: int = field(default_factory=person_count.__next__)
    # deventer: bool

    def __hash__(self):
        return id(self)
    def __eq__(self, other):
        return id(self) == id(other)
    def __lt__(self, other):
        return id(self) < id(other)
    def __str__(self):
        return f"{self.name} {'EN' if self.english else 'NL'} {'F' if self.female else 'M'} {self.current_class if self.current_class else '?'} {sorted(self.interests)}"


def mutate_solution(solution):
    solution = solution.copy()

    i1, i2, i3 = sample(range(len(solution)), 3)

    solution[i1] = solution[i1].copy()
    solution[i2] = solution[i2].copy()

    j1 = randrange(len(solution[i1]))
    j2 = randrange(len(solution[i2]))
    
    if randrange(2)==0:
        # swap two
        solution[i1][j1], solution[i2][j2] = solution[i2][j2], solution[i1][j1]
    else:
        # swap three
        solution[i3] = solution[i3].copy()
        j3 = randrange(len(solution[i3]))
        solution[i1][j1], solution[i2][j2], solution[i3][j3] = solution[i2][j2], solution[i3][j3], solution[i1][j1]
    return solution


def crossover_solutions(solution1, solution2):
    # print_solution(solution1)
    # print_solution(solution2)

    if False:
        # Take groups from both at random
        solution = solution1 + solution2
        shuffle(solution)
    else:
        # Take a single random group from 2, and try to maintain groups from 1 as much as possible
        solution = [choice(solution2)] + solution1

    solution = [group.copy() for group in solution]
    remove_duplicates(solution)
    solution = merge_groups(solution, solution1)
    # print_solution(solution)
    return solution


def print_solution(solution):
    people = sum((len(group) for group in solution))
    print(f"groups={len(solution)} people={people}")
    for group in solution:
        print(f"- {group[0]}")
        for person in group[1:]:
            print(f"  {person}")
            

def remove_duplicates(solution):
    seen = set()
    for group in solution:
        person_index = 0
        while person_index < len(group):
            person = group[person_index]
            if person in seen:
                group.pop(person_index)
            else:
                seen.add(person)
                person_index += 1


def merge_groups(solution, org_solution):
    solution = sorted(solution, key=lambda x: -len(x))
    for index1 in range(len(org_solution)):
        group_size = len(org_solution[index1])
        while len(solution[index1]) + len(solution[-1]) <= group_size:
            for index2 in range(index1+1, len(solution)):
                if len(solution[index1]) + len(solution[index2]) <= group_size:
                    solution[index1] += solution.pop(index2)
                    break
            else:
                break
        while len(solution[index1]) < group_size:
            solution[index1].append(solution[-1].pop())
            if len(solution[-1]) == 0:
                solution.pop()
    assert len(solution) == len(org_solution)
    return solution


def calculate_fitness(solution):
    score = 0
    for group in solution:
        english = 0
        female = 0
        current_class = None
        current_students = 0
        interests = []
        for person in group:
            if person.english:
                english += 1
            if person.female:
                female += 1
            if person.current_class:
                current_students += 1
                if current_class:
                    if current_class != person.current_class:
                        current_class = "mixed"
                else:
                    current_class = person.current_class
            interests += person.interests
        if current_class != "mixed":
            score += 1000 # no mixed-class groups please!

        score += 800 * (current_students / len(group) - 0.5) # max 400 points when 50/50 old/new
        score += 200 if female==0 or female==2 else 100 if female==3 else 0
        score += 200 if english==0 or english==2 else 100 if english==3 else 0
        uniq_interest_count = len(set(interests))
        dup_interest_count = len(interests) - uniq_interest_count
        score += dup_interest_count / len(interests) * 500 # 500 * percentage of shared interests

    return round(score)


def generate_fake_people(count):
    people = []
    for id in range(count):
        people.append(Person(
            name = f"#{id}",
            english = randrange(7)==0,
            female = randrange(7)==0,
            # deventer = randrange(7)==0,
            current_class = None if randrange(2)==0 else choice(["a", "b"]),
            interests = sample(INTERESTS, 3),
        ))
    return people


def get_random_solution(solutions):
    index = min(randrange(len(solutions)), randrange(len(solutions)))
    # index = randrange(len(solutions))
    return solutions[index]


def generate_random_solution(people):
    group_count = math.ceil(len(people) / MAX_GROUP_SIZE)
    group_members = len(people) / group_count
    ordered_people = sample(people, len(people))
    solution = [ordered_people[int(group_members*group_index):int(group_members*(group_index+1))] for group_index in range(group_count)]
    return solution


def main():
    seed(5)
    people = generate_fake_people(90)
    # Best seen fitness for this data set: 26250!
    seed()
    generation = [[generate_random_solution(people) for _ in range(POOL_SIZE)] for _ in range(POOL_COUNT)]

    gen_count = 0
    last_progress = [0 for _ in range(POOL_COUNT)]
    fitnesses = [0 for _ in range(POOL_COUNT)]
    while True:
        gen_count += 1
        pool_solutions = []
        for pool_num, pool in enumerate(generation):
            results = []
            for solution in pool:
                fitness = calculate_fitness(solution)
                results.append((-fitness, solution))
            results.sort()

            best_fitness = -results[0][0]
            if best_fitness > fitnesses[pool_num]:
                fitnesses[pool_num] = best_fitness
                last_progress[pool_num] = 0
            else:
                last_progress[pool_num] += 1
            pool_solutions.append([result for fitness, result in results[0:POOL_SIZE//3]])

        new_generation = []
        for pool_num, solutions in enumerate(pool_solutions):
            if pool_num == 0:
                new_pool = [s[0] for s in pool_solutions]
            elif last_progress[pool_num] >= STALL_GENERATIONS:
                # new_pool = [generate_random_solution(people) for _ in range(POOL_SIZE)]
                new_pool = [choice(pool_solutions)[0], solutions[0]]
                last_progress[pool_num] = 0
                fitnesses[pool_num] = 0
            else:
                new_pool = [solutions[0]]
            while len(new_pool) < POOL_SIZE:
                mode = randrange(10) 
                if mode < 5:
                    new_solution = crossover_solutions(get_random_solution(solutions), get_random_solution(solutions))
                elif mode < 10:
                    new_solution = mutate_solution(get_random_solution(solutions))
                else:
                    new_solution = get_random_solution(solutions)
                new_pool.append(new_solution)

            new_generation.append(new_pool)

        print(f"Generation #{gen_count} fitness={fitnesses}")
        if gen_count%100 == 0:
            print_solution(pool_solutions[0][0])
        generation = new_generation


if __name__ == '__main__':
    main()
