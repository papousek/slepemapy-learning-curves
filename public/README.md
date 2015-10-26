# slepemapy.cz

The data set is based on the online system http://slepemapy.cz. The system is
available in Czech, English, German, and Spanish, most users are from Czech Republic
(84 %) and Slovakia (8 %). The system uses adaptive algorithms for choosing
questions, these algorithms are described in detail in [1] and [2].

This is the first publicly available data set collected to evaluate the system,
it is static and caputers users' interactions from 25 August to 24 October
2015. Results and description of studied conditions are available in [3].
The basic statistics of the data set are as follows:

  - 20,389 students;
  - 1,683 geographical items
  - 1,346,568 answers

## Description

The dataset contains 2 CSV files (commas are used as delimiter):

 - answers of users practicing location of places;
 - users' feedback about difficulty.

### Answers

|        Column       | Description                                                                                              |
|:-------------------:|----------------------------------------------------------------------------------------------------------|
|          id         | answer identifier                                                                                        |
|         user_id     | user's identifier                                                                                        |
|     item_asked_id   | identifier of the asked place                                                                            |
|     term_asked_name | name of the asked place                                                                                  |
|    item_answered_id | identifier of the answered place, empty if the user answered "I don't know"                              |
|     term_asked_name | name of the asked place                                                                                  |
|       term_type     | type of the practiced place                                                                              |
|      context_name   | name of the practiced map (context)                                                                       |
|        direction    | type of the answer: (t2d) find the given place on the map; (d2t) pick the name for the highlighted place |
|        options      | number of options (the asked place included)                                                             |
|          time       | datetime when the answer was inserted to the system                                                      |
|     response_time   | how much time the answer took (measured in milliseconds)                                                 |
|      condition      | identifier of the studied condtion for which the asked question was constructed                          |
|      ip_country     | country retrieved from the user’s IP address                                                             |
|        ip_id        | meaningless identifier of the user’s IP address                                                          |

### Feedback

|        Column       | Description                                                                                              |
|:-------------------:|----------------------------------------------------------------------------------------------------------|
|          id         | identifier                                                                                               |
|        user_id      | user's identifier                                                                                        |
|        inserted     | datetime when the record was inserted to the system                                                      |
|         value       | user's feedback about difficulty                                                                         |


## Ethical and privacy considerations:

The used educational system is used mainly by students in schools or by
students preparing for exams. Nevertheless, it is an open online system which
can be used by anybody and details about individual users are not available.
Users are identified only by their anonymous ID. Users can log into the system
using their Google or Facebook accounts; but this login is used only for
identifying the user within the system, it is not included in the data set.
Unlogged users are tracked using web browser cookies. The system also logs IP
address from which users access the system, the IP address is included in the
data set in anonymized form. We separately encode the country of origin, which
can be useful for analysis and its inclusion is not a privacy concern. The rest
of the IP address is replaced by meaningless identifier to preserve privacy.

## Terms of Use

The data set is available at http://www.fi.muni.cz/adaptivelearning/data/slepemapy/

### License

This data set is made available under Open Database License whose full text can
be found at http://opendatacommons.org/licenses/odbl/. Any rights in individual
contents of the database are licensed under the Database Contents License whose
text can be found http://opendatacommons.org/licenses/dbcl/

### Citation

Please cite the following paper when you use our data set:

```
TODO
```


## References

 - **[1]** Papoušek, J., Pelánek, R. & Stanislav, V. Adaptive Practice of Facts in Domains with Varied Prior Knowledge. In Educational Data Mining, 2014.
 - **[2]** Papoušek, J., & Pelánek, R. Impact of adaptive educational system behaviour on student motivation. In Artificial Intelligence in Education, 2015.
 - **[3]** TODO
