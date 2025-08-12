from manim import *


class HelloWorld(Scene):
    def construct(self):
        text = Text("AI-Powered Academic Explainer")
        self.play(Write(text))
        self.wait(2)
