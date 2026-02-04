"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { QUESTIONS, SECTIONS, type Section, TOTAL_QUESTIONS, type ClosedQuestion } from "@/lib/models/assessments/questions";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";

interface QuestionnaireProps {
  onComplete: (answers: Record<string, number | string>) => void;
  onSkip?: () => void;
}

const SECTION_LABELS: Record<Section, { title: string; description: string; emoji: string }> = {
  psychological: {
    title: "Personalità",
    description: "Come ti comporti e reagisci nelle situazioni",
    emoji: "🧠",
  },
  values: {
    title: "Valori",
    description: "Cosa è importante per te nella vita",
    emoji: "💎",
  },
  interests: {
    title: "Interessi",
    description: "I tuoi hobby e passioni",
    emoji: "🎯",
  },
  behavioral: {
    title: "Relazioni",
    description: "Come ti comporti nelle relazioni",
    emoji: "💕",
  },
};

export function Questionnaire({ onComplete, onSkip }: QuestionnaireProps) {
  const [currentSectionIndex, setCurrentSectionIndex] = useState(0);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number | string>>({});
  const [showingSectionIntro, setShowingSectionIntro] = useState(true);

  const currentSection = SECTIONS[currentSectionIndex];
  const questions = QUESTIONS[currentSection];
  const currentQuestion = questions[currentQuestionIndex];

  // Calculate progress
  const totalAnswered = Object.keys(answers).length;
  const progress = (totalAnswered / TOTAL_QUESTIONS) * 100;

  const handleClosedAnswer = (value: number) => {
    setAnswers((prev) => ({
      ...prev,
      [currentQuestion.id]: value,
    }));

    // Auto-advance after selection
    setTimeout(() => goNext(), 300);
  };

  const handleOpenAnswer = (text: string) => {
    setAnswers((prev) => ({
      ...prev,
      [currentQuestion.id]: text,
    }));
  };

  const goNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex((prev) => prev + 1);
    } else if (currentSectionIndex < SECTIONS.length - 1) {
      setCurrentSectionIndex((prev) => prev + 1);
      setCurrentQuestionIndex(0);
      setShowingSectionIntro(true);
    } else {
      // Completed!
      onComplete(answers);
    }
  };

  const goPrev = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex((prev) => prev - 1);
    } else if (currentSectionIndex > 0) {
      setCurrentSectionIndex((prev) => prev - 1);
      const prevSectionQuestions = QUESTIONS[SECTIONS[currentSectionIndex - 1]];
      setCurrentQuestionIndex(prevSectionQuestions.length - 1);
      setShowingSectionIntro(false);
    }
  };

  const currentAnswer = answers[currentQuestion?.id];
  const canGoNext = currentAnswer !== undefined && currentAnswer !== "";
  const isFirstQuestion = currentSectionIndex === 0 && currentQuestionIndex === 0;
  const isLastQuestion = currentSectionIndex === SECTIONS.length - 1 && currentQuestionIndex === questions.length - 1;

  // Section intro screen
  if (showingSectionIntro) {
    const sectionInfo = SECTION_LABELS[currentSection];
    return (
      <div className="space-y-6">
        <div className="text-center mb-8">
          <Progress value={progress} className="h-2 mb-4" />
          <p className="text-sm text-muted-foreground">
            Sezione {currentSectionIndex + 1} di {SECTIONS.length}
          </p>
        </div>

        <Card className="text-center">
          <CardHeader>
            <div className="text-6xl mb-2">{sectionInfo.emoji}</div>
            <CardTitle className="text-2xl">{sectionInfo.title}</CardTitle>
            <CardDescription>{sectionInfo.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-6">
              {questions.length} domande
            </p>
            <Button onClick={() => setShowingSectionIntro(false)} size="lg">
              Inizia
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Get scale labels for current question
  const scaleLabels = currentQuestion.type === "closed"
    ? (currentQuestion as ClosedQuestion).scaleLabels
    : ["1", "5"];

  return (
    <div className="space-y-6">
      {/* Progress */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-muted-foreground">
            {SECTION_LABELS[currentSection].emoji} {SECTION_LABELS[currentSection].title}
          </span>
          <span className="text-sm text-muted-foreground">
            {totalAnswered}/{TOTAL_QUESTIONS}
          </span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {/* Question Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-medium leading-relaxed">
            {currentQuestion.text}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {currentQuestion.type === "closed" ? (
            <div className="space-y-4">
              {/* Scale labels */}
              <div className="flex justify-between text-sm text-muted-foreground px-2">
                <span>{scaleLabels[0]}</span>
                <span>{scaleLabels[1]}</span>
              </div>

              {/* 1-5 Scale with clickable dots */}
              <div className="flex items-center justify-between gap-2 py-2">
                {[1, 2, 3, 4, 5].map((value) => {
                  const isSelected = currentAnswer === value;
                  return (
                    <button
                      key={value}
                      onClick={() => handleClosedAnswer(value)}
                      className="flex flex-col items-center gap-2 group flex-1"
                    >
                      <div
                        className={`
                          w-12 h-12 rounded-full border-2 flex items-center justify-center
                          transition-all duration-200 cursor-pointer mx-auto
                          ${isSelected
                            ? "border-primary bg-primary text-primary-foreground scale-110"
                            : "border-muted-foreground/30 bg-background text-muted-foreground hover:border-primary/50 hover:bg-primary/10"
                          }
                        `}
                      >
                        <span className="text-lg font-semibold">{value}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <Textarea
                value={(currentAnswer as string) || ""}
                onChange={(e) => handleOpenAnswer(e.target.value)}
                placeholder={currentQuestion.placeholder}
                className="min-h-32"
              />
              <p className="text-xs text-muted-foreground">
                Scrivi liberamente, più dettagli condividi meglio sarà il matching.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          onClick={goPrev}
          disabled={isFirstQuestion}
        >
          <ChevronLeftIcon className="w-4 h-4 mr-1" />
          Indietro
        </Button>

        {currentQuestion.type === "open" && (
          <Button onClick={goNext} disabled={!canGoNext}>
            {isLastQuestion ? "Completa" : "Avanti"}
            {!isLastQuestion && <ChevronRightIcon className="w-4 h-4 ml-1" />}
          </Button>
        )}
      </div>

      {/* Skip option */}
      {onSkip && (
        <div className="text-center">
          <button
            onClick={onSkip}
            className="text-sm text-muted-foreground hover:text-foreground underline"
          >
            Salta per ora (potrai completare dopo)
          </button>
        </div>
      )}
    </div>
  );
}
