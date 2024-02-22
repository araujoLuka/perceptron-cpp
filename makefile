parametrosCompilacao=-Wall -O3 -msse3
nomePrograma=perceptron
# all cpp files except main.cpp
objetos=$(patsubst %.cpp,%.o,$(filter-out main.cpp,$(wildcard *.cpp)))

all: $(nomePrograma)

$(nomePrograma): $(objetos) main.cpp
	g++ -o $(nomePrograma) $? $(parametrosCompilacao)

%.o: %.cpp
	g++ -c $< $(parametrosCompilacao)

debug: parametrosCompilacao+=-DDEBUG -g
debug: all

clean:
	rm -f *.o

purge: clean
	rm -f $(nomePrograma)
