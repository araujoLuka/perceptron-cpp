parametrosCompilacao=-Wall
nomePrograma=perceptron
objetos=$(patsubst %.cpp, %.o, $(wildcard *.cpp))

all: $(nomePrograma)

$(nomePrograma): $(objetos)
	g++ -o $(nomePrograma) $(objetos) $(parametrosCompilacao)

%.o: %.cpp
	g++ -c $< $(parametrosCompilacao)

clean:
	rm -f *.o

purge: clean
	rm -f $(nomePrograma)
