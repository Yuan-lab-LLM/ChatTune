import { ClipboardListIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";

const ResourceAuditPanel = () => {
  const { t } = useTranslation();
  const audit = trpc.listResourceAuditEvents.useQuery();
  const events = audit.data?.data || [];

  return (
    <section className="rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm">
      <div className="mb-4">
        <div className="flex items-center gap-2">
          <ClipboardListIcon className="size-4 text-primary" />
          <h3 className="text-base font-semibold">{t("resourceAccess.auditLogTitle")}</h3>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          {t("resourceAccess.auditLogDescription")}
        </p>
      </div>

      <div className="rounded-xl border border-border/45 bg-muted/15 p-3">
        <div className="grid max-h-[52vh] gap-1 overflow-y-auto text-xs text-muted-foreground">
          {events.map((event) => (
            <div key={event.id}>
              {new Date(event.createdAt).toLocaleString()} · {event.eventType} · {event.actorUserId || t("resourceAccess.systemActor")}
              {event.resourceId ? ` · ${event.resourceId}` : ""}
            </div>
          ))}
          {!events.length && <div>{t("resourceAccess.noAuditRecords")}</div>}
        </div>
      </div>
    </section>
  );
};

export default ResourceAuditPanel;
