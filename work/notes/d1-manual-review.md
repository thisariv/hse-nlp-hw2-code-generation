---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-15
updated: 2026-10-15
sources:
  - nlp-hw02-d1-results
aliases: []
---

# D1: ручная проверка 30 кандидатов

Собственный разбор условия и кода; **прокси**, не независимая эталонная разметка.
Выборка:30 случайных без возвращения из всех38 644 кандидатов A,
NumPy default_rng, seed2026101501; row ids отсортированы только для показа.
Условия не подбирались по содержанию или доступности дублей. В выборке нет строк C.

**Reference ошибочен в14/30 (46,7%)**, включая одно обычное прочтение обмена
в №3; консервативно без него13/30 (43,3%). Модели правы в9/30 (30,0%),
консервативно8/30. Ещё5 случаев имеют ошибочный reference и неверный/неподтверждённый
модельный ответ. Reference содержательно верен в4/30; в1/30 только округление.
В9/30 условие неполно/противоречиво; ещё2/30 допускают разные трактовки.
Эти11 случаев не записаны в подтверждённые ошибки reference.

Исходный D1 сохранил **ответы и статусы, но не программы или токены моделей**.
Ниже исходный target восстановлен из pinned compact-cache, а модельный код —
**новый CPU FP32 greedy replay** обеих pinned L3 FP16 weights на frozen D1 decoder.
Все60 ответов replay совпали с сохранёнными D1; все30 reference переисполнены
и совпали. Это делает код пригодным для содержательной проверки ответа,
но не доказывает совпадение исходных T4 токенов. Модельный код не становится train-target.

## Примеры

> [!example]- 1. row 56340: Неполное или противоречивое условие
> 
> In 4 years, the sum of Danny's age and Ellie's age will be 43. Ellie is currently twice as old as Danny was 8 years ago. If Danny is currently 15 years old, how old will Ellie be in 5 years?
> 
> **Вердикт:** Условия противоречат друг другу: из возраста Danny15 получается Ellie14, сумма через4 года37, а не43. По условию суммы получилось бы Ellie20. Единственного ответа нет; модель19 использует лишь второе условие, reference15 не обоснован.
> 
> **Reference → 15**
> ```python
> def solve() -> int:
>     Danny_in_8yrs_ago = 15-8
>     Ellie = Danny_in_8yrs_ago*2
>     Danny_in_4yrs = 15+4
>     EplusDin4yrs = 43
>     Ellie_in_5yrs = EplusDin4yrs - Danny_in_4yrs - Ellie + 5
>     result = Ellie_in_5yrs
>     return result
> ```
> 
> **C-main → 19** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     danny_age = 15
>     ellie_age = 2 * (danny_age - 8)
>     ellie_age_in_4_years = 43 - (danny_age + 4)
>     ellie_age_in_5_years = ellie_age + 5
>     result = ellie_age_in_5_years
>     return result
> ```
> 
> **C-long2x → 19** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     danny_age = 15
>     ellie_age = 2 * (danny_age - 8)
>     ellie_age_in_5_years = ellie_age + 5
>     result = ellie_age_in_5_years
>     return result
> ```

> [!example]- 2. row 539334: Reference неверен; модели правы
> 
> A baby was born at 5:12 pm and it is now 11:00 pm. How many hours old is the baby?
> 
> **Вердикт:** С17:12 до23:00 проходит5ч48мин =5,8ч. Reference игнорирует12 минут.
> 
> **Reference → 6**
> ```python
> def solve() -> int:
>     hour1 = 17
>     hour2 = 23
>     hours_old = hour2 - hour1
>     result = f'<<<{hours_old}>>>'
>     result = hours_old
>     return result
> ```
> 
> **C-main → 5.8** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     birth_time = 5.2
>     current_time = 11
>     hours = current_time - birth_time
>     result = hours
>     return result
> ```
> 
> **C-long2x → 5.8** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     birth_time = 5.2
>     current_time = 11.0
>     hours = current_time - birth_time
>     result = hours
>     return result
> ```

> [!example]- 3. row 833875: Reference неверен; модели правы
> 
> Emma has 50 marbles. She exchanges 2/5 of her marbles with her friend. Then, she receives 6 marbles from her father. Finally, she gives away 1/4 of the total marbles to her cousins. How many marbles does Emma have left?
> 
> **Вердикт:** При обычном прочтении exchanges как «отдала» остаётся (50−20+6)·3/4=27. Reference вместо оставшихся30 берёт переданные20. Формулировка обмена не задаёт встречную передачу; это условный вердикт, отдельно учтён в консервативной оценке.
> 
> **Reference → 19.5**
> ```python
> def solve() -> int:
>     marbles= 50
>     marbles_given= marbles * 2 / 5
>     marbles_received= marbles_given + 6
>     marbles_remaining= marbles_received * 3/4
>     result= marbles_remaining
>     return result
> ```
> 
> **C-main → 27** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     marbles = 50
>     marbles -= marbles * 2/5
>     marbles += 6
>     marbles -= marbles * 1/4
>     result = marbles
>     return result
> ```
> 
> **C-long2x → 27** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 4. row 1382003: Неполное или противоречивое условие
> 
> Then he cycles four times as far as he cycled on Monday every Thursday and Friday. How many miles will he cycle in a month with 4 weeks?
> 
> **Вердикт:** Начало условия потеряно: нет расстояния в понедельник и полного недельного расписания. Reference подставляет14, модель10; оба числа взяты вне условия.
> 
> **Reference → 392**
> ```python
> def solve() -> int:
>     result = 0
>     for week in range(4):
>       result = result + 14 * 3 + 14 * 4
>     return result
> ```
> 
> **C-main → 80** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     miles_per_day = 10
>     days_per_week = 7
>     weeks_per_month = 4
>     miles_per_week = miles_per_day * 2
>     miles_per_month = miles_per_week * weeks_per_month
>     result = miles_per_month
>     return result
> ```
> 
> **C-long2x → 80** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     miles_per_day = 10
>     miles_per_week = miles_per_day * 2
>     miles_per_month = miles_per_week * 4
>     result = miles_per_month
>     return result
> ```

> [!example]- 5. row 1624384: Reference неверен; модели также не дают буквального ответа
> 
> It takes Brandon 40 minutes to build one set of shelves. If he builds 5 sets of shelves over both Saturday and Sunday, how many hours does he spend on building them?
> 
> **Вердикт:** Буквально5 комплектов за выходные:5·40/60=10/3ч. Если имелось в виду5 в каждый день, получается20/3ч модели. Reference8ч не соответствует ни одному прочтению: добавляет ещё80мин. Модель не подтверждена буквальным условием.
> 
> **Reference → 8**
> ```python
> def solve() -> int:
>     time_for_one_set_shelves = 40/1
>     total_time_for_both_days = (time_for_one_set_shelves * 5) * 2
>     account_for_minutes = (40/1) * 2
>     total_time_in_hours = (total_time_for_both_days+account_for_minutes)/60
>     result = total_time_in_hours
>     return result
> ```
> 
> **C-main → 6.66666666667** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     sets_per_day = 5
>     minutes_per_set = 40
>     days_per_week = 2
>     total_sets = sets_per_day * days_per_week
>     total_minutes = total_sets * minutes_per_set
>     total_hours = total_minutes / 60
>     result = total_hours
>     return result
> ```
> 
> **C-long2x → 6.66666666667** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     time_per_set = 40
>     sets_per_day = 5
>     days = 2
>     total_time = time_per_set * sets_per_day * days
>     hours = total_time / 60
>     result = hours
>     return result
> ```

> [!example]- 6. row 1780108: Reference неверен; модели также не дают буквального ответа
> 
> am has 20 coupons that he can use to buy video games. Each coupon is worth $5. He also has $40 in cash. How many video games can he buy if each game costs $15?
> 
> **Вердикт:** Денег20·5+40=140; можно купить floor(140/15)=9 целых игр. Reference9⅓ не учитывает целочисленность; модель вычитает стоимость купонов из наличных и даёт−4.
> 
> **Reference → 9.33333**
> ```python
> def solve() -> int:
>     g = 140 / 15
>     result = g
>     return result
> ```
> 
> **C-main → -4** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     coupons = 20
>     coupon_value = 5
>     cash = 40
>     game_cost = 15
>     total_coupon_value = coupons * coupon_value
>     remaining_cash = cash - total_coupon_value
>     games = remaining_cash // game_cost
>     result = games
>     return result
> ```
> 
> **C-long2x → -4** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 7. row 1919857: Reference неверен; модели правы
> 
> Lucas is having a party. He ordered enough pizza for 14 people. If 8 people came to the party, how much pizza is left over?
> 
> **Вердикт:** Остаются порции для14−8=6 человек. Reference (14/8)−14 не имеет нужного смысла.
> 
> **Reference → -12.25**
> ```python
> def solve() -> int:
>     leftover_pizza = (14/8) - 14
>     result = leftover_pizza
>     return result
> ```
> 
> **C-main → 6** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     total_pizza = 14
>     people = 8
>     pizza_left = total_pizza - people
>     result = pizza_left
>     return result
> ```
> 
> **C-long2x → 6** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     people = 14
>     pizza = 1
>     people -= 8
>     pizza_left = pizza * people
>     result = pizza_left
>     return result
> ```

> [!example]- 8. row 1929301: Округление reference
> 
> A box contains 6 red balls and 4 blue balls. If two balls are chosen at random, what is the probability that both balls are red?
> 
> **Вердикт:** Без возвращения вероятность6/10·5/9=1/3. Reference вычисляет ту же формулу и намеренно округляет до4 знаков. Это численное расхождение допуска1e−6, а не содержательная ошибка решения; не считаю реальной ошибкой reference.
> 
> **Reference → 0.3333**
> ```python
> def solve() -> int:
>     result = round((6/10)*(5/9), 4)
>     return result
> ```
> 
> **C-main → 0.333333333333** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     total_balls = 10
>     red_balls = 6
>     probability_first_red = red_balls / total_balls
>     probability_second_red = (red_balls - 1) / (total_balls - 1)
>     probability_both_red = probability_first_red * probability_second_red
>     result = probability_both_red
>     return result
> ```
> 
> **C-long2x → 0.333333333333** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 9. row 2038200: Reference прав; модели ошиблись
> 
> At a party, there are 5 boys and 7 girls. Each boy gives a candy to every girl and each girl gives a candy to every boy. How many candies are exchanged in total?
> 
> **Вердикт:** В каждую сторону5·7 передач: всего70 конфет. Модели складывают число участников и дают12.
> 
> **Reference → 70**
> ```python
> def solve() -> int:
>     total = 35 + 35
>     result = total
>     return result
> ```
> 
> **C-main → 12** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     boys = 5
>     girls = 7
>     candies = boys + girls
>     result = candies
>     return result
> ```
> 
> **C-long2x → 12** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     boys = 5
>     girls = 7
>     total_candies = boys + girls
>     result = total_candies
>     return result
> ```

> [!example]- 10. row 2119680: Неоднозначное условие
> 
> Jake has four times more bottles than Laura. If Jake has 16 bottles, how many bottles does Laura have?
> 
> **Вердикт:** Four times more допускает обычное «в4 раза» (4 бутылки) и буквальное «на400% больше» (3,2). Reference использует второе, модели первое; без соглашения о языке нельзя объявлять reference ошибкой.
> 
> **Reference → 3.2**
> ```python
> def solve() -> int:
>     Bottles_Laura = 16 / 5
>     result = 3.2
>     return result
> ```
> 
> **C-main → 4** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     jake_bottles = 16
>     laura_bottles = jake_bottles / 4
>     result = laura_bottles
>     return result
> ```
> 
> **C-long2x → 4** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 11. row 2219509: Reference прав; модели ошиблись
> 
> In a race, Sarah runs at twice the speed of Jack. If they both take a total of 45 minutes to run, how long did it take Jack to finish the race?
> 
> **Вердикт:** Для одной дистанции время Jack вдвое больше Sarah: t+t/2=45, t=30мин. Модели возвращают время Sarah15мин.
> 
> **Reference → 30**
> ```python
> def solve() -> int:
>     result = 30
>     return result
> ```
> 
> **C-main → 15** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     total_time = 45
>     sarah_speed = 2
>     jack_speed = 1
>     jack_time = total_time / (sarah_speed + jack_speed)
>     result = jack_time
>     return result
> ```
> 
> **C-long2x → 15** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 12. row 2356451: Reference неверен; модели также не дают буквального ответа
> 
> A triangular prism has a base with sides 8m, 10m, and 12m. The height of the prism is 15m. What is the volume of the prism?
> 
> **Вердикт:** По Герону полупериметр15, площадь sqrt(15·7·5·3)=15sqrt(7). Объём225sqrt(7)≈595,294. Reference25312,5 и модели225 используют неверную площадь.
> 
> **Reference → 25312.5**
> ```python
> def solve() -> int:
>     base_area = 1/4 * (8 + 10 + 12) * (15)**2
>     volume = base_area * 15
>     result = round(volume, 2)
>     return result
> ```
> 
> **C-main → 225** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     base_side1 = 8
>     base_side2 = 10
>     base_side3 = 12
>     height = 15
>     base_area = (base_side1 + base_side2 + base_side3) / 2
>     volume = base_area * height
>     result = volume
>     return result
> ```
> 
> **C-long2x → 225** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     base_side1 = 8
>     base_side2 = 10
>     base_side3 = 12
>     height = 15
>     base_area = 0.5 * (base_side1 + base_side2 + base_side3)
>     volume = base_area * height
>     result = volume
>     return result
> ```

> [!example]- 13. row 2419026: Reference неверен; модели правы
> 
> If it takes 5 people 2 hours to paint a fence, how many hours will it take for 3 people to paint the same fence?
> 
> **Вердикт:** Работа10 человеко-часов; троим нужно10/3ч. Reference умножает время на число работников вместо обратной пропорции.
> 
> **Reference → 6**
> ```python
> def solve() -> int:
>     total_effort = 5 * 2
>     effort_per_person = total_effort / 5
>     time_needed = effort_per_person * 3
>     result = time_needed
>     return result
> ```
> 
> **C-main → 3.33333333333** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     people1 = 5
>     time1 = 2
>     people2 = 3
>     time2 = (people1*time1)/people2
>     result = time2
>     return result
> ```
> 
> **C-long2x → 3.33333333333** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     people1 = 5
>     hours1 = 2
>     people2 = 3
>     hours2 = (people1 * hours1) / people2
>     result = hours2
>     return result
> ```

> [!example]- 14. row 2591265: Reference неверен; модели правы
> 
> If you add 5 years to six times Kate's age, the result is 56. How old is Kate?
> 
> **Вердикт:** 6x+5=56, x=8,5. Аннотация ->int не требует обрезать математический возраст; int в reference теряет полгода.
> 
> **Reference → 8**
> ```python
> def solve() -> int:
>     divide_by = 6
>     sum_result = 56
>     kate_age = int((sum_result - 5) / divide_by)
>     result = kate_age
>     return result
> ```
> 
> **C-main → 8.5** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     kate_age = (56 - 5) / 6
>     result = kate_age
>     return result
> ```
> 
> **C-long2x → 8.5** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 15. row 3809793: Reference неверен; модели правы
> 
> John has a garden with 30 rose bushes. Each rose bush has 6 flowers. If only half of the flowers bloom, how many roses does John have in total?
> 
> **Вердикт:** Цветущих роз30·6/2=90. Reference смешивает количество кустов и цветков и получает45.
> 
> **Reference → 45**
> ```python
> def solve() -> int:
>     rose_bush_count = 30
>     flower_count = rose_bush_count * 6
>     blooming_flowers = flower_count // 2
>     john_new_rose_count =rose_bush_count + blooming_flowers // 6
>     result = john_new_rose_count
>     return result
> ```
> 
> **C-main → 90** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     rose_bushes = 30
>     flowers_per_bush = 6
>     blooming_flowers = (rose_bushes * flowers_per_bush) / 2
>     result = blooming_flowers
>     return result
> ```
> 
> **C-long2x → 90** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     rose_bushes = 30
>     flowers_per_bush = 6
>     total_flowers = rose_bushes * flowers_per_bush
>     blooming_flowers = total_flowers / 2
>     result = blooming_flowers
>     print(f'<<<{result}>>>')
>     return result
> ```

> [!example]- 16. row 3866446: Reference прав; модели ошиблись
> 
> In a bakery, there are twice as many croissants as muffins. If there are 45 baked goodies in total, what is the number of muffins available in the bakery?
> 
> **Вердикт:** Если маффинов x, круассанов2x, то3x=45 и x=15. Модели переставляют категории и отвечают30. Необычные tuple-переменные reference не мешают фактическому numeric15.
> 
> **Reference → 15**
> ```python
> def solve() -> int:
>     croissants = ()
>     muffins = ()
>     total_baked_goodies = 45
>     croissants = 2 * muffins
>     muffins = total_baked_goodies / 3
>     result = muffins
>     return result
> ```
> 
> **C-main → 30** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     total_baked_goods = 45
>     croissants = total_baked_goods / 3
>     muffins = total_baked_goods - croissants
>     result = muffins
>     return result
> ```
> 
> **C-long2x → 30** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 17. row 4020725: Reference прав; модели ошиблись
> 
> The angles in a convex quadrilateral are in a ratio of 3:5:6:7. What is the measure of the largest angle?
> 
> **Вердикт:** У четырёхугольника сумма360°, максимальный угол360·7/21=120°. Модели используют180° и дают60°.
> 
> **Reference → 120**
> ```python
> def solve() -> int:
>     ratio = [3, 5, 6, 7]
>     sum_of_ratios = sum(ratio)
>     num_of_angles = 4
>     total_degree_sum = (num_of_angles - 2) * 180
>     largest_angle = ratio[-1] / sum_of_ratios * (total_degree_sum)
>     result = largest_angle
>     return result
> ```
> 
> **C-main → 60** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     ratio_sum = 3 + 5 + 6 + 7
>     largest_angle = 7 / ratio_sum * 180
>     result = largest_angle
>     return result
> ```
> 
> **C-long2x → 60** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     ratio_sum = 3 + 5 + 6 + 7
>     largest_angle = (7 / ratio_sum) * 180
>     result = largest_angle
>     return result
> ```

> [!example]- 18. row 4265800: Reference неверен; модели правы
> 
> Mina trains for a 5k race and runs at a pace of 7 minutes per kilometer. If she trains five days a week and runs for thirty minutes each day, how many total kilometers did she run each week?
> 
> **Вердикт:** За неделю150мин; при7мин/км расстояние150/7≈21,4286км. Reference смешивает часы и минуты и дополнительно умножает на5.
> 
> **Reference → 1.78571**
> ```python
> def solve() -> int:
>     days = 5
>     time = 30.0/60
>     pace = 7.0
>     distance = days*(time/pace)*5
>     result = distance
>     return result
> ```
> 
> **C-main → 21.4285714286** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     distance_per_day = 30 / 7
>     distance_per_week = distance_per_day * 5
>     result = distance_per_week
>     return result
> ```
> 
> **C-long2x → 21.4285714286** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     pace = 7
>     training_days = 5
>     running_time = 30
>     total_distance = (running_time / pace) * training_days
>     result = total_distance
>     return result
> ```

> [!example]- 19. row 4540250: Неоднозначное условие
> 
> Farmer Brown wants to buy 15 bales of the better quality hay from Chester. Each bale costs $18. If Farmer Brown only has $250, how many more bales can he afford?
> 
> **Вердикт:** На250 долларов можно купить13 тюков, до желаемых15 не хватает2. Reference2 отвечает на разумное исправленное «сколько не хватает»; буквальное «сколько ещё может купить» не задаёт исходной покупки. Модель−2 неверна; reference удалять по согласию моделей нельзя.
> 
> **Reference → 2**
> ```python
> def solve() -> int:
>     desired_delivery = 15
>     available_funds = 250
>     cost_per_bale = 18
>     max_affordable_delivery = available_funds // cost_per_bale
>     remaining_delivery = desired_delivery - max_affordable_delivery
>     result = remaining_delivery
>     return result
> ```
> 
> **C-main → -2** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     total_cost = 15 * 18
>     remaining_money = 250 - total_cost
>     remaining_bales = remaining_money // 18
>     result = remaining_bales
>     return result
> ```
> 
> **C-long2x → -2** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     bale_cost = 18
>     total_bales = 15
>     total_cost = bale_cost * total_bales
>     remaining_money = 250 - total_cost
>     remaining_bales = remaining_money // bale_cost
>     result = remaining_bales
>     return result
> ```

> [!example]- 20. row 5576677: Неполное или противоречивое условие
> 
> Emily wants to buy a new camera and decides to go to a photography store. The camera costs $800 and Emily's credit card offers extended warranty protection for electronic purchases. What is the capacity of Emily's memory card?
> 
> **Вердикт:** Стоимость камеры и гарантия не определяют ёмкость карты памяти. Reference выдумывает35/50 и деление на10, модели подставляют800.
> 
> **Reference → 88.5**
> ```python
> def solve() -> int:
>     memory_card_price = 35
>     extended_warranty_price = 50
>     camera_price = 800
>     total_price = camera_price + memory_card_price
>     total_price += extended_warranty_price
>     result = total_price/10
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-main → 800** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     price_of_camera = 800
>     price_of_memory_card = 0
>     price_of_memory_card = price_of_camera
>     result = price_of_memory_card
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-long2x → 800** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 21. row 5991325: Неполное или противоречивое условие
> 
> Riley and Max went to the park. Riley brought 3 sandwiches and Max brought half as many sandwiches. How many books has Ada read in the past year?
> 
> **Вердикт:** Сэндвичи Riley/Max не содержат сведений о книгах Ada. Ни7,875, ни4,5 не следуют из условия.
> 
> **Reference → 7.875**
> ```python
> def solve() -> int:
>     Riley_brought = 3
>     Max_brought = 3/2
>     total_sandwiches = Riley_brought + Max_brought
>     Ada_books_last_year = (total_sandwiches/2)*(total_sandwiches-1)
>     result = Ada_books_last_year
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-main → 4.5** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     sandwiches_riley = 3
>     sandwiches_max = sandwiches_riley / 2
>     books_ada = sandwiches_riley + sandwiches_max
>     result = books_ada
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-long2x → 4.5** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     riley_sandwiches = 3
>     max_sandwiches = riley_sandwiches / 2
>     ada_books = riley_sandwiches + max_sandwiches
>     result = ada_books
>     print(f'<<<{result}>>>')
>     return result
> ```

> [!example]- 22. row 6124558: Неполное или противоречивое условие
> 
> During the school talent show, Michael performed a magic trick and impressed everyone with his skills. His best friend, Emma, watched the show and applauded loudly. What is the shoe size of Emma?
> 
> **Вердикт:** Размер обуви Emma не дан и не выводится из фокуса и аплодисментов. Обе стороны выдумывают размер.
> 
> **Reference → 12**
> ```python
> def solve() -> int:
>     shoe_size_of_michael = 10
>     friend_of_michael = "Emma"
>     plaudits = 'loudly'
>     if friend_of_michael == "Emma" and plaudits == 'loudly':
>         shoe_size_of_emma = shoe_size_of_michael + 2
>     else:
>         shoe_size_of_emma = shoe_size_of_michael - 1
>     result = shoe_size_of_emma
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-main → 10** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     shoe_size = 10
>     result = shoe_size
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-long2x → 10** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 23. row 6563351: Reference неверен; модели правы
> 
> Sarah is planning to go for a hike that takes 3 hours. She also wants to spend 1 hour at the park before heading home. If Sarah needs to be back home by 6:00 pm, what is the latest time she can start the hike?
> 
> **Вердикт:** При отсутствии дополнительного времени на дорогу начать нужно в14:00, то есть2pm:18−3−1=14. Модели2 используют12-часовую запись. Reference смешивает60мин и18часов и даёт−60.
> 
> **Reference → -60**
> ```python
> def solve() -> int:
>     park_time = 60
>     return_home_time = 18
>     total_time_needed = park_time + return_home_time
>     latest_hike_start_time = return_home_time - total_time_needed
>     result = latest_hike_start_time
>     return result
> ```
> 
> **C-main → 2** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     hours_hiking = 3
>     hours_park = 1
>     total_hours = hours_hiking + hours_park
>     latest_start_time = 6 - total_hours
>     result = latest_start_time
>     return result
> ```
> 
> **C-long2x → 2** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     hike_duration = 3
>     park_duration = 1
>     total_time = hike_duration + park_duration
>     latest_start_time = 6 - total_time
>     result = latest_start_time
>     return result
> ```

> [!example]- 24. row 7239421: Reference неверен; модели правы
> 
> A bus travels at a speed of 60km/h for half the journey and 40km/h for the remaining journey. If the total distance to be covered is 150km, what is the average speed of the bus during the entire journey?
> 
> **Вердикт:** Для двух половин расстояния по75км:150/(75/60+75/40)=48км/ч. Reference неверно переводит расстояния во время.
> 
> **Reference → 26.6667**
> ```python
> def solve() -> int:
>     time_taken = ((75*2) / 60) + ((75*2.5) / 60)
>     distance = 150
>     average_speed = distance / time_taken
>     result = average_speed
>     return result
> ```
> 
> **C-main → 48** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     distance = 150
>     speed1 = 60
>     speed2 = 40
>     time1 = distance / (2 * speed1)
>     time2 = distance / (2 * speed2)
>     total_time = time1 + time2
>     average_speed = distance / total_time
>     result = average_speed
>     return result
> ```
> 
> **C-long2x → 48** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     distance = 150
>     speed1 = 60
>     speed2 = 40
>     time1 = distance / 2 / speed1
>     time2 = distance / 2 / speed2
>     total_time = time1 + time2
>     average_speed = distance / total_time
>     result = average_speed
>     return result
> ```

> [!example]- 25. row 7327167: Reference неверен; модели также не дают буквального ответа
> 
> A family enjoys eating pizza every night. They usually order a 14-inch pizza for the three of them, but they want to try different sizes. If they want to eat the same amount of pizza, how many 10-inch pizzas will they need to order instead of their usual one 14-inch pizza?
> 
> **Вердикт:** Количество эквивалентных по площади пицц (14/10)^2=1,96, либо2 целых при заказе не меньше исходной площади. Reference1 и модели1,4 неверны.
> 
> **Reference → 1**
> ```python
> def solve() -> int:
>     usual_pizza_size = 14
>     new_pizza_size = 10
>     ratio_of_areas = (new_pizza_size/2) ** 2 / (usual_pizza_size/2) **2
>     pizzas_needed = round(ratio_of_areas * 1)
>     result = pizzas_needed
>     return result
> ```
> 
> **C-main → 1.4** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     usual_size = 14
>     new_size = 10
>     num_pizzas = usual_size / new_size
>     result = num_pizzas
>     return result
> ```
> 
> **C-long2x → 1.4** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     size_of_14_inch_pizza = 14
>     size_of_10_inch_pizza = 10
>     result = size_of_14_inch_pizza / size_of_10_inch_pizza
>     return result
> ```

> [!example]- 26. row 7617448: Reference неверен; модели также не дают буквального ответа
> 
> Sheila's friend2 was impressed with her determination to save money and decided to match every dollar she saves. If her friend2 matches her monthly savings of $276, how much money will be in Sheila's piggy bank after 4 years?
> 
> **Вердикт:** Взнос отца удваивает месячные276:2·276·12·4=26496 при нулевом неуказанном начальном балансе. Reference выдумывает ещё3000+7000; модели забывают взнос отца.
> 
> **Reference → 36496**
> ```python
> def solve() -> int:
>     current_savings = 3000
>     monthly_saving = 276
>     monthly_saving_dad_matched = monthly_saving * 2
>     duration = 4
>     additional_savings = 7000
>     total_savings = current_savings + (monthly_saving_dad_matched * 12 * duration) + additional_savings
>     result = total_savings
>     return result
> ```
> 
> **C-main → 13248** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     savings_per_month = 276
>     months_in_a_year = 12
>     years = 4
>     total_savings = savings_per_month * months_in_a_year * years
>     result = total_savings
>     return result
> ```
> 
> **C-long2x → 13248** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     savings_per_month = 276
>     months_per_year = 12
>     years = 4
>     total_savings = savings_per_month * months_per_year * years
>     result = total_savings
>     return result
> ```

> [!example]- 27. row 7854986: Неполное или противоречивое условие
> 
> Sarah is planning a road trip and wants to divide her travel itinerary into equal segments. She plans to drive the same distance each day for 7 days. However, she realizes that she needs to travel 40 miles more each day compared to the previous day. If her total travel distance is 350 miles, how many miles does she plan to drive on the first day?
> 
> **Вердикт:** Одинаковая дистанция каждый день противоречит прибавке40. Даже если оставить арифметическую прогрессию, первый член350/7−3·40=−70 миль: физически невозможен. Reference−30 и модели−190 не исправляют условие.
> 
> **Reference → -30**
> ```python
> def solve() -> int:
>     days_total = 7
>     travel_distance = 350
>     previous_day_distance = (travel_distance - (days_total*(days_total-1)*20)) / days_total
>     first_day_distance = previous_day_distance + 40
>     result = first_day_distance
>     return result
> ```
> 
> **C-main → -190** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     total_distance = 350
>     days = 7
>     distance_per_day = total_distance / days
>     first_day_distance = distance_per_day - (days - 1) * 40
>     result = first_day_distance
>     return result
> ```
> 
> **C-long2x → -190** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 28. row 8441972: Неполное или противоречивое условие
> 
> A bookshelf has twice as many books as another bookshelf, which has 30 books. After 10 books were taken from the smaller bookshelf, the bigger bookshelf turned up with 50 books more than the smaller bookshelf. How many books were originally on the bigger bookshelf?
> 
> **Вердикт:** Первое условие даёт большой стеллаж60 книг; после удаления10 из малого разность40, а не50. Reference70 использует только разность, модели110 добавляют50 к исходным60. Все условия несовместимы.
> 
> **Reference → 70**
> ```python
> def solve() -> int:
>     smaller = 30 - 10
>     bigger = 50 + smaller
>     smaller = smaller * 2
>     result = bigger
>     return result
> ```
> 
> **C-main → 110** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     smaller_bookshelf = 30
>     bigger_bookshelf = 2 * smaller_bookshelf
>     smaller_bookshelf -= 10
>     bigger_bookshelf += 50
>     result = bigger_bookshelf
>     return result
> ```
> 
> **C-long2x → 110** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     books_small = 30
>     books_big = 2 * books_small
>     books_small -= 10
>     books_big += 50
>     result = books_big
>     return result
> ```

> [!example]- 29. row 9213450: Неполное или противоречивое условие
> 
> Liam is a cashier who stands for 8 hours every day. If Liam weighs 160 pounds and walks for 30 minutes during his lunch break, how many calories does he burn standing for 8 hours?
> 
> **Вердикт:** Не дана модель расхода энергии или соответствующий коэффициент. Вес и время не определяют единственное число калорий. Reference вводит эмпирическую формулу, модели100ккал/ч; оба добавляют внешние предположения.
> 
> **Reference → 378**
> ```python
> def solve() -> int:
>     import math 
>     timeStanding = 8*60
>     calPerMin = 0.0175 * math.pow(160, 0.75)
>     calBurned = timeStanding * calPerMin
>     result = math.ceil(calBurned)
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-main → 800** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     hours_standing = 8
>     calories_burned_per_hour = 100
>     calories_burned = hours_standing * calories_burned_per_hour
>     result = calories_burned
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-long2x → 800** (CPU replay; D1 ответ совпал)
> Тот же код, что у C-main.

> [!example]- 30. row 9578661: Неполное или противоречивое условие
> 
> Lily is going for a hike. She climbed a steep hill and then took a photo. How many kilometers did Natalie ride in total?
> 
> **Вердикт:** В условии Lily и фото; спрашивается расстояние Natalie без чисел и связи с рассказом. Reference6 и модели15 выдуманы.
> 
> **Reference → 6**
> ```python
> def solve() -> int:
>     import math
>     hill_distance = 3
>     hill_distance *= 2
>     distance_to_hill = hill_distance / 2
>     remaining_distance = distance_to_hill
>     total_distance = distance_to_hill + remaining_distance
>     result = total_distance
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-main → 15** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     hill_distance = 10
>     photo_distance = 5
>     total_distance = hill_distance + photo_distance
>     result = total_distance
>     print(f'<<<{result}>>>')
>     return result
> ```
> 
> **C-long2x → 15** (CPU replay; D1 ответ совпал)
> ```python
> def solve() -> int:
>     distance_hike = 10
>     distance_photo = 5
>     total_distance = distance_hike + distance_photo
>     result = total_distance
>     print(f'<<<{result}>>>')
>     return result
> ```

## Источники

- [d1-results](d1-results.md) и [d1-plan](d1-plan.md) — собственный протокол, SHA моделей и границы вывода.
- [Вердикты и sampling receipt](../results/d1-manual-review.json), [рендер отчёта](../research/d1-manual-report.py), [CPU replay](../research/d1-results.py).
- Локальный ignored `results/d1-labels/manual-review-data.json`: восстановленные программы, ответы и execution receipts; SHA в sampling receipt.
- Библиография: nlp-hw02-d1-results.
