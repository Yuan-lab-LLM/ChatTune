import { Button, Spin } from "antd";
import {
  ArrowRight,
  BookOpen,
  Compass,
  MessageSquare,
  Rocket,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";
import { useFirstTimeGuide } from "@/context/FirstTimeGuideContext";

const UserLandingPage = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const { startTour } = useFirstTimeGuide();
  const serviceQuery = trpc.getSharedServiceAvailability.useQuery();
  const targetQuery = trpc.getLatestRunnableRun.useQuery();
  const target = targetQuery.data?.data;
  const available = Boolean(serviceQuery.data?.data?.available && target);

  const enterChat = (mode?: "training" | "practice") => {
    if (!target) return;
    if (mode === "training") {
      sessionStorage.setItem("medflow_open_training_guide", "true");
    }
    if (mode === "practice") {
      sessionStorage.setItem("medflow_open_quick_start", "true");
    }
    navigate(`/projects/${target.project}/runs/${target.runId}`);
  };

  const startPageTour = () => {
    if (!target) return;
    startTour(target.project);
  };

  if (serviceQuery.isLoading || targetQuery.isLoading) {
    return <div className="flex h-full items-center justify-center"><Spin /></div>;
  }

  const guideCards = [
    {
      step: "01",
      icon: Compass,
      iconClassName: "bg-sky-50 text-sky-600 ring-sky-100 dark:bg-sky-500/10 dark:text-sky-300 dark:ring-sky-500/20",
      title: t("home.landing.getStarted.pageTour.title"),
      description: t("home.landing.getStarted.pageTour.description"),
      action: t("userLanding.pageTourAction"),
      onClick: startPageTour,
    },
    {
      step: "02",
      icon: BookOpen,
      iconClassName: "bg-teal-50 text-teal-600 ring-teal-100 dark:bg-teal-500/10 dark:text-teal-300 dark:ring-teal-500/20",
      title: t("home.landing.getStarted.textGuide.title"),
      description: t("home.landing.getStarted.textGuide.description"),
      action: t("userLanding.trainingAction"),
      onClick: () => enterChat("training"),
    },
    {
      step: "03",
      icon: Rocket,
      iconClassName: "bg-orange-50 text-orange-600 ring-orange-100 dark:bg-orange-500/10 dark:text-orange-400 dark:ring-orange-500/20",
      title: t("home.landing.getStarted.practice.title"),
      description: t("home.landing.getStarted.practice.description"),
      action: t("userLanding.practiceAction"),
      onClick: () => enterChat("practice"),
    },
  ];

  return (
    <div className="h-full overflow-auto bg-[linear-gradient(180deg,_#eef6ff_0%,_#f8fbff_42%,_#f3f7fb_100%)] px-4 py-6 dark:bg-[linear-gradient(180deg,_#07111f_0%,_#0f172a_48%,_#111827_100%)] sm:px-6 lg:px-10">
      <div className="mx-auto flex min-h-full max-w-6xl flex-col justify-center gap-5 py-4 lg:gap-6">
        <section className="overflow-hidden rounded-lg border border-white/80 bg-white/92 shadow-[0_22px_70px_-44px_rgba(15,23,42,0.34)] dark:border-white/10 dark:bg-slate-950/76">
          <div className="min-h-[330px]">
            <div className="flex flex-col justify-between gap-9 p-7 sm:p-9 lg:p-11">
              <div className="space-y-6">
                <div className="flex flex-wrap items-center gap-3">
                  <div
                    className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs font-medium ${
                      available
                        ? "border-emerald-200/80 bg-emerald-50/90 text-emerald-700 dark:border-emerald-500/20 dark:bg-emerald-500/10 dark:text-emerald-300"
                        : "border-amber-200/80 bg-amber-50/90 text-amber-700 dark:border-amber-500/20 dark:bg-amber-500/10 dark:text-amber-300"
                    }`}
                  >
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        available ? "bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,0.14)]" : "bg-amber-500"
                      }`}
                    />
                    {available ? t("userLanding.available") : t("userLanding.unavailable")}
                  </div>
                </div>

                <div className="max-w-3xl">
                  <h1 className="text-3xl font-semibold leading-tight text-slate-950 dark:text-slate-50 sm:text-4xl lg:text-[46px]">
                    {t("userLanding.title")}
                  </h1>
                  <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-300 sm:text-base">
                    {t("userLanding.description")}
                  </p>
                </div>
              </div>

              <div className="flex justify-end">
                <Button
                  type="primary"
                  size="large"
                  icon={<MessageSquare className="h-4 w-4" />}
                  disabled={!available}
                  onClick={() => enterChat()}
                  className="h-11 rounded-lg border-0 bg-[linear-gradient(180deg,#1687d9_0%,#0d6fbd_100%)] px-5 font-medium shadow-[0_18px_32px_-24px_rgba(13,111,189,0.72)]"
                >
                  {t("userLanding.start")}
                </Button>
              </div>
            </div>
          </div>
        </section>

        <div className="grid gap-4 md:grid-cols-3">
          {guideCards.map(({ step, icon: Icon, iconClassName, title, description, action, onClick }) => (
            <button
              key={title}
              type="button"
              disabled={!available}
              onClick={onClick}
              className="group flex min-h-[210px] flex-col rounded-lg border border-white/80 bg-white/88 p-5 text-left shadow-[0_18px_46px_-34px_rgba(15,23,42,0.34)] transition duration-200 hover:-translate-y-0.5 hover:border-sky-200 hover:shadow-[0_24px_52px_-34px_rgba(14,116,184,0.34)] disabled:cursor-not-allowed disabled:opacity-55 dark:border-white/10 dark:bg-slate-950/62 dark:hover:border-sky-500/30"
            >
              <div className="mb-6 flex items-center justify-between">
                <div className={`flex h-11 w-11 items-center justify-center rounded-lg ring-1 ${iconClassName}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <span className="text-xs font-semibold text-slate-300 dark:text-slate-600">{step}</span>
              </div>
              <h2 className="text-lg font-semibold text-slate-950 dark:text-slate-50">{title}</h2>
              <p className="mt-2 flex-1 text-sm leading-6 text-muted-foreground">{description}</p>
              <div className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-sky-700 transition group-hover:gap-3 dark:text-sky-300">
                {action}
                <ArrowRight className="h-4 w-4" />
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default UserLandingPage;



