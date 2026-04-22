#include <AD9833.h>
#include <digitalWriteFast.h>

constexpr long int start = 1000;
constexpr long int end = 3500000;
constexpr long int step = 17500;
long int currentFreq;

int lastTime = 0;;
int counter = 0;

AD9833 gen(10);

void setup() {
  gen.Begin();
  gen.ApplySignal(SINE_WAVE, REG0, start);
  gen.EnableOutput(true);
  currentFreq = start;
  Serial.begin(9600);
}

void loop() {
  //float deltaTime = millis() - lastTime;
  //lastTime = millis();
  //counter += deltaTime;

  gen.ApplySignal(SINE_WAVE, REG0, currentFreq);
  delay(10);
  Serial.print("(");
  Serial.print(currentFreq);
  Serial.print(",");
  Serial.print(analogRead(A0));
  Serial.print(")\n");
  currentFreq += step;
  if (currentFreq > end) {
    currentFreq = start;
  }
  //Serial.print("A0:");
  //Serial.print(analogRead(A0));
  //Serial.print(",");
  //Serial.print("A1:");
  //Serial.print(analogRead(A1));
  //Serial.print(",");
  //Serial.print(",");
  //Serial.print("A3:");
  //Serial.println(analogRead(A2));
}
