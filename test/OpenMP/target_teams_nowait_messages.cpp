// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

int main(int argc, char **argv) {
  #pragma omp target teams nowait( // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();
  #pragma omp target teams nowait (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();
  #pragma omp target teams nowait device (-10u)
  foo();
  #pragma omp target teams nowait (3.14) device (-10u) // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();

  return 0;
}
