# Gestión de recursos humanos e investigación de aprendizaje automático: mi experiencia
----

#Translations:
* [English](README.md)
* [Español](README-es.md)
There is no translation on plots titles.

<br/>
<p align = "left">
<img src = "img / iit.JPG" alt = "iit">
</p> <br/>

## Introducción
Este artículo está dividido en dos partes. El primero es mi experiencia en el desarrollo de una Revisión sistemática de literatura en el Instituto Indio de Tecnología de Delhi. El segundo es lo que me pareció importante en el campo de los recursos humanos con respecto a la inteligencia artificial. Y proporciono un ejemplo utilizando un conjunto de fechas Kaggle para el riesgo de desgaste en cualquier empresa (si un empleado se va o no).

## Instituto Indio de Tecnología de Delhi
Estar en la India ha sido una de las mejores experiencias que he tenido. Estoy muy orgulloso de ser parte e hice grandes amigos en una de las universidades más prestigiosas de la India: el Instituto Indio de Tecnología de Delhi, IIT Delhi. También cuéntale a todos mis nuevos amigos sobre mi gran universidad en México y sobre México en general.
Cuando llegué, me di cuenta de que no todos hablaban inglés en la India, casi todas las personas entienden (creo que sí), pero hablar por ellos es difícil. Cuando llegué a Satpura, lo que llamé house, un estudiante casi graduado, Dhruv, me ayudó a activar mi SIM y trató de arreglar mi estancia en Satpura, complicada. Más tarde, mi amigo y mano derecha, Swapnil hizo todo lo posible para obtener mi habitación: SA-10 en la planta baja, mucho mejor en el suelo, allí, no estaba tan caliente. ¡El primer día fue la cosa! Sin cadenas, colchones muy livianos e incómodos y muchos mosquitos me despertaron a las 5 de la mañana.
¡La aventura comenzó! La comida en el desorden era muy buena, al menos para mí, como extranjero me pareció genial. Sin embargo, algunos amigos me dijeron que comer el mismo alimento durante cuatro años es un infierno.
No conocía a nadie en la India en ese momento (excepto por el Prof. Vigneswara, Swapnil y Drhuv), y no tenía clases. Así que me encontré con gente en el desorden o en la sala de televisión. Uno de los primeros amigos que hice en la India fue Jeevarej de Tamil Nadu, quien hizo su postdoctorado en matemáticas, trabajando en lógica difusa (más tarde me di cuenta de lo importante que es para el modelado). Solo le pregunté, "¿hablas inglés?" Me dijo que sí, y empezamos a hablar, también me dijo que no habla hindi, así que cuando llegó a IIT Delhi, estaba haciendo la misma pregunta.
Delhi es muy hermosa, tan diversa y llena de patrimonio, está en el corazón de la India, de hecho, eso es lo que Delh significa en inglés: CORAZÓN. Si estás allí debes visitar AGRA y JAIPUR: EL TRIÁNGULO DE ORO.
<br/>
<p align = "center">
<img src = "img / dhruv.JPG" alt = "dhruv">
</p> <br/>
<br/>
<p align = "center">
<img src = "img / jeev.JPG" alt = "jeev">
</p> <br/>

## Cultura india.
Disfruté el cine, la música, la comida y el arte. India está llena de vegetarianos, sin embargo, puedes comer pollo. Comí pollo masala, pani puri, Palak Paneer, Malai Kofta, Kaali Daal, Chole Bhature, Aloo Ka Halwa, Paratha, Lassi dulce, Curry, pollo Biriyani, Jalebi, Gulab Jamun, Idili, Melu Vada, Masala Dosa, Tandori Chicken, Chapati
<br/>
<p align = "center">
<img src = "img / cur_pp.png" alt = "food">
</p> <br/>
Las películas están llenas de emociones. Todos los que estudian en una escuela de ingeniería en la India han visto "3 IDIOTS" es una gran película, ¡muy real! PK, Barfi !, Namastey London, también fueron muy buenas películas. Fui a un musical de Bollywood: ZANGOORA, en El reino de los sueños.
Disfruté visitando Delhi, Agra y Jaipur, sus templos, sus mercados y todos los sitios.
Visité muchos lugares, desde el famoso Taj Mahal en Agra hasta un antiguo mercado de libros dominical en Delhi.
<br/>
<p align = "center">
<img src = "img / qutab_minar_book.png" alt = "qutab_minar_book">
</p> <br/>


## Reglas de decisión en un contexto HRM
Después de leer alrededor de 150 artículos, descubrí dos cosas importantes, no hay definiciones claras con respecto a la extracción de datos, la ciencia de la información, el aprendizaje automático, etc. En segundo lugar, la importancia de comprender los algoritmos y el uso de las reglas de decisión.
La ciencia de datos es una disciplina amplia, aunque el concepto es reciente, cada día evoluciona.
> Según la Escuela de Información de Berkeley, el Ciclo de Vida de la Ciencia de Datos tiene cinco etapas, estas etapas no son exclusivas unas de otras. Estas cinco etapas son captura de datos, mantenimiento de datos, proceso de datos, análisis de datos y comunicación de datos.
Esta última es la actividad más importante en las empresas. Es donde ofrecemos visualizaciones de datos, informes de datos, inteligencia de negocios y toma de decisiones. La mayoría de las veces, todo el tiempo dedicado a las etapas restantes terminará en la toma de decisiones, en función de todo el proceso, por lo que podemos mejorar constantemente.
Al tratar de resolver un problema de ciencia de datos, hay muchas técnicas que puede utilizar para descifrarlo; por ejemplo, máquinas de vectores de soporte (SVM), árboles de decisión, regresión logística, redes neuronales y muchos otros. El gran problema es que casi todas son cajas negras, probablemente sabrás lo que hace el algoritmo, ojalá, pero tus compañeros de trabajo o personas comunes y corrientes podrían no entender la intuición detrás de esos modelos complejos. Entonces, como científico de datos, no solo debe extraer valor de los datos, sino también poder traducir los resultados en soluciones y luego comunicarlos. Las reglas de decisión nos ayudarán a extraer conclusiones claras sobre cómo el algoritmo está tomando decisiones. Así que más tarde podemos tomar decisiones basadas en eso.
Podemos extraer reglas de decisión utilizando Python y algunas bibliotecas conocidas. Vamos a utilizar un conjunto de datos de recursos humanos proporcionado por el concurso Kaggle. Las características del conjunto de datos son el nivel de satisfacción, la última evaluación, el número de proyecto, el promedio de horas mensuales, el tiempo empleado en la empresa, el accidente laboral, la izquierda, la promoción de los últimos 5 años, las ventas y el salario. La variable <strong> Izquierda </strong> será nuestro objetivo. Básicamente, queremos que se extraigan dos cosas de nuestro algoritmo, la predicción de nuestra variable objetivo, en este caso si el empleado se fue o no y por qué se está yendo. ¿Qué características aumenta el riesgo de desgaste? Esas características serán el resto del conjunto de datos, las variables que utilizaremos para alimentar el árbol de decisiones. Básicamente, el nodo raíz es nuestro conjunto de datos que luego se dividirá en función de nuestra estrategia seleccionada como el Índice de Gini, Chi-cuadrado, entropía o reducción de la varianza. El conjunto de datos tiene 10 columnas y 14,999 observaciones. 2 características son tipos de datos de objetos, para trabajar con scikit-learn transformaremos en dummy esas características, usando pandas. Usaremos un tren y una prueba de datos; La prueba tendrá el 35% de los datos completos.
Para hacer que el árbol sea más fácil de interpretar, estableceremos la profundidad máxima del árbol en 5, para que podamos tener muestras más representativas de todos los nodos. La puntuación sigue siendo bastante buena con .97, y podemos generalizar las reglas proporcionadas por el árbol con mayor facilidad. Como en todo, enfrentaremos la compensación, entre una mayor precisión y un algoritmo más fácil de interpretar.
Una vez que llamemos a todos los métodos, obtendremos un archivo de puntos, podemos abrir ese archivo con un procesador de texto. No te preocupes, no es tan complicado. Esencialmente, hay un montón de pasos para construir un árbol de decisión visual, también puedes hacer eso con python. ¡Pero es aún más fácil visualizar todo ese código, simplemente usando la web! Copie y pegue todas las líneas en: <br/> http://www.webgraphviz.com/ <br/> Luego, haga clic en <strong> Generar gráfico </strong>.
<br/>
<p align = "left">
<img src = "img / webgraph.PNG" alt = "webgraph">
</p>
Obtendrá automáticamente un árbol de decisión. Si es robusto, probablemente podría considerar mover los parámetros del algoritmo.

## ¿Listo?
<br/>
<p align = "center">
<img src = "img / decision_tree.PNG" alt = "dec_tree">
</p>
Permite extraer las reglas del árbol de decisión. Primero algunos conceptos: <br/>

* Entropía:
En otras palabras, la medida de la imprevisibilidad del contenido de la información es la cantidad de información que aprendemos en promedio de una instancia. Buscamos cero cuando hablamos de entropía. Cero significa que solo hay una etiqueta en el nodo.

* Muestras:
La cantidad de observaciones en cada nodo.

* Valor:
La cantidad de observaciones en cada etiqueta. El que está a la izquierda es el valor cero y el que está a la derecha es el número 1. En este caso, a la izquierda, respectivamente.

Para extraer las reglas de decisión de nuestro árbol debemos considerar los nodos al final (nodos finales) y luego hasta el nodo raíz (hacia atrás) o viceversa (hacia adelante). Es importante cubrir toda la rama. De lo contrario, nuestras decisiones serán menos precisas.
Para extraer conclusiones relevantes, debemos ponderar esos nodos finales con toneladas de observaciones sin importar que sean 0 o 1 (adentro o izquierda). Las más relevantes incluyen 1039, 981, 5261, 631 muestras. <br/>
1. Si el promedio de horas trabajadas es de más de 126 horas y el resultado de la última evaluación está entre 0,445 y 0,574, y tiene menos de 2 proyectos y un nivel de satisfacción menor de 0,465 estará más inclinado a irse la organización.
<br/>
<p align = "center">
<img src = "img / dec_1039.PNG" alt = "dec_tree">
</p>
2. Si el nivel de satisfacción de los trabajadores es inferior a 0,465 y el promedio de horas trabajadas mensuales es inferior a 290 horas y, finalmente, el número de proyectos es de 3 a 6, el empleado pasará a permanecer en la organización.
<br/>
<p align = "left">
<img src = "img / dec_981_5261.PNG" alt = "dec2">
</p>
3. Si el trabajador tiene entre 3 y 5 proyectos y trabajó menos de 290 horas y tiene menos de 4 años en la organización y un nivel de satisfacción de más de 0.465, estarán más dispuestos a permanecer en la empresa. <br/>
4. Si el tiempo que se pasa en la organización es de entre 5 y 7 años y la puntuación de la última evaluación es más de 0.8 puntos, trabajando más de 214 horas en promedio al mes, con un nivel de satisfacción de más de 0.46, serán propensos a abandonar el empresa.
<br/>
<p align = "right">
<img src = "img / dec_631.PNG" alt = "dec3">
</p> <br/>

Finalmente, trazar la importancia de las funciones puede ser una buena idea. Esto nos puede dar una idea de lo que un trabajador valora más o, al menos, lo que más se preocupa. Puede obtener esta información a través del parámetro feature_importances_. Y luego solo hay que trazarlo.
Definitivamente, el nivel de satisfacción es muy importante al tomar una decisión, también los años en la empresa y la última evaluación, por último, pero no menos importante, la cantidad de proyectos y las horas mensuales de trabajo. Podemos decir que los accidentes en la organización, promociones, salario y cargo son irrelevantes.
<br/>
<p align = "center">
<img src = "img / depth_of_the_features.png" alt = "i_f">
</p>

## Conclusiones
Contratar a un empleado es muy costoso y despedir a un empleado es aún más costoso (también hay que tener en cuenta todos los costos de contratación). Es posible que no se consideren todas las reglas extraídas con este conjunto de datos para tomar decisiones en una empresa. Este conjunto de datos kaggle se simula, pero es muy útil para visualizar cómo extraer información de datos empresariales y convertirla en políticas reales. Esta información proporcionada por Kaggle está casi limpia, por lo que el proceso fue completamente fácil. Cuando trabaje en una base de datos real (especialmente en HRM), sus datos pueden ser muy desordenados, así que tenga en cuenta que podría dedicar aproximadamente el 80% de su tiempo a limpiar los datos. No se asuste de estadísticas simples, siempre son útiles y fáciles de entender para todos. Siempre recuerde que saber por qué elige una decisión en particular es muy importante, es por eso que estamos usando reglas de deicisión. Una vez que sepamos por qué se van, podemos motivarlos en consecuencia, nos vamos a un mundo a medida.