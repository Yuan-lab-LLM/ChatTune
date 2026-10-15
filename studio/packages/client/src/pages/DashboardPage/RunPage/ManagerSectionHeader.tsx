import { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { ManagementCacheMeta } from "@shared/types";
import { useAuth } from "@/context/AuthContext";
import CacheMetaInfo from "./CacheMetaInfo";

interface Props {
  title: string;
  count?: number;
  actions?: ReactNode;
  guideAction?: ReactNode;
  cacheMeta?: ManagementCacheMeta | null;
  stackActions?: boolean;
}

const ManagerSectionHeader = ({
  title,
  count,
  actions,
  guideAction,
  cacheMeta,
  stackActions = false,
}: Props) => {
  const { t } = useTranslation();
  const { isAdmin } = useAuth();
  const shouldShowContainer = !isAdmin && Boolean(cacheMeta?.containerName);

  return (
    <div className="mb-0 shrink-0 space-y-1.5">
      <div className={stackActions ? "flex flex-col gap-2" : "flex items-center justify-between gap-3"}>
        <h3 className={`${stackActions ? "w-full" : "min-w-0 flex-1 truncate"} text-[13px] font-semibold uppercase tracking-[0.06em] text-foreground`}>
          <span className="align-middle">{title}</span>
          {typeof count === "number" ? (
            <span className="ml-1.5 whitespace-nowrap text-[11px] text-muted-foreground">
              ({count})
            </span>
          ) : null}
        </h3>
        {actions ? (
          <div className={`${stackActions ? "w-full" : "shrink-0"} flex items-center justify-end gap-1.5`}>
            {actions}
          </div>
        ) : null}
      </div>

      <div className="rounded-2xl border border-border/20 bg-muted/15 px-3 py-1.5">
        <div className="flex items-center justify-between gap-3">
          {shouldShowContainer ? (
            <p className="text-[11px] text-muted-foreground">
              {t("query.current-container")}:{" "}
              <span className="font-medium">{cacheMeta?.containerName}</span>
            </p>
          ) : null}
          {guideAction}
        </div>
        {cacheMeta ? <CacheMetaInfo meta={cacheMeta} /> : null}
      </div>
    </div>
  );
};

export default ManagerSectionHeader;
