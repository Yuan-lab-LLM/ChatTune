import { useMemo, useState } from "react";
import { Share2Icon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";
import { Button } from "@/components/ui/button";
import { translateAuthError } from "@/utils/authErrors";

type ResourceSharingPanelProps = {
  embedded?: boolean;
};

type CatalogResource = {
  id: string;
  bizType?: string | null;
  nodeId?: string | null;
  containerName?: string | null;
  itemKey?: string | null;
  visibility?: string | null;
};

type PublicationRequest = {
  id: string;
  resourceId: string;
  requesterUserId: string;
  requesterUsername?: string | null;
  reviewedBy?: string | null;
  reviewerUsername?: string | null;
  reviewNote?: string | null;
  status: string;
  createdAt?: string | Date;
  updatedAt?: string | Date;
  resource?: CatalogResource | null;
};

type PublicationHistoryItem = {
  id: string;
  kind?: string;
  status: string;
  resourceId?: string | null;
  requesterUserId?: string | null;
  requesterUsername?: string | null;
  actorUserId?: string | null;
  actorUsername?: string | null;
  reviewerUserId?: string | null;
  reviewerUsername?: string | null;
  note?: string | null;
  createdAt?: string | Date;
  updatedAt?: string | Date;
  resource?: CatalogResource | null;
};

const valueText = (value?: string | number | Date | null) =>
  value === null || value === undefined || value === "" ? "-" : String(value);

const shortId = (value?: string | null) =>
  value && value.length > 12 ? `${value.slice(0, 8)}...${value.slice(-4)}` : value || "-";

const formatDateTime = (value?: string | Date | null) => {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
};

const statusClassName = (status: string, kind?: string) => {
  if (kind === "direct_publish") return "border-sky-200 bg-sky-50 text-sky-700";
  if (status === "approved") return "border-emerald-200 bg-emerald-50 text-emerald-700";
  if (status === "rejected") return "border-rose-200 bg-rose-50 text-rose-700";
  return "border-amber-200 bg-amber-50 text-amber-700";
};

const ResourceSharingPanel = ({ embedded = false }: ResourceSharingPanelProps) => {
  const { t } = useTranslation();
  const catalog = trpc.listResourceCatalog.useQuery();
  const review = trpc.reviewResourcePublication.useMutation();
  const [message, setMessage] = useState("");

  const resources = (catalog.data?.data.resources || []) as CatalogResource[];
  const requests = (catalog.data?.data.requests || []) as PublicationRequest[];
  const resourceById = useMemo(
    () => new Map(resources.map((resource) => [resource.id, resource])),
    [resources],
  );
  const pendingRequests = requests.filter((request) => request.status === "pending");
  const publicationHistory = ((catalog.data?.data.publicationHistory || requests) as PublicationHistoryItem[])
    .filter((item) => item.kind === "direct_publish" || item.status !== "pending")
    .sort((left, right) =>
      new Date(right.updatedAt || right.createdAt || 0).getTime() -
      new Date(left.updatedAt || left.createdAt || 0).getTime(),
    );

  const refresh = async () => {
    await catalog.refetch();
  };

  const reviewRequest = async (requestId: string, approved: boolean) => {
    setMessage("");
    try {
      await review.mutateAsync({ requestId, approved });
      setMessage(approved ? t("resourceAccess.publicationApproved") : t("resourceAccess.publicationRejected"));
      await refresh();
    } catch (error) {
      setMessage(translateAuthError(error, t, "resourceAccess.publicationReviewFailed"));
    }
  };

  const sectionClassName = embedded
    ? ""
    : "rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm";
  const historyStatusLabel = (item: PublicationHistoryItem) => {
    if (item.kind === "direct_publish") return t("resourceAccess.publicationPublishedDirectly");
    return t(`resourceAccess.publicationStatus.${item.status}`, { defaultValue: item.status });
  };
  const requestResource = (request: PublicationRequest) =>
    request.resource || resourceById.get(request.resourceId) || null;
  const historyResource = (item: PublicationHistoryItem) =>
    item.resource || (item.resourceId ? resourceById.get(item.resourceId) || null : null);

  return (
    <section className={sectionClassName}>
      {!embedded && (
        <div className="mb-4">
          <div className="flex items-center gap-2">
            <Share2Icon className="size-4 text-primary" />
            <h3 className="text-base font-semibold">{t("resourceAccess.sharingAndPublicationTitle")}</h3>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("resourceAccess.sharingAndPublicationDescription")}
          </p>
        </div>
      )}

      {message && (
        <div className="mb-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">
          {message}
        </div>
      )}

      <div className="grid gap-5">
        <div>
          <h4 className="mb-2 text-sm font-semibold">{t("resourceAccess.publicationPendingTitle")}</h4>
          <div className="grid gap-3">
            {pendingRequests.map((request) => {
              const resource = requestResource(request);
              return (
                <div key={request.id} className="rounded-xl border border-border/45 bg-background p-3">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <div className="text-sm font-medium">{valueText(resource?.itemKey || request.resourceId)}</div>
                      <div className="text-xs text-muted-foreground">
                        {valueText(resource?.bizType)} · {valueText(resource?.nodeId)} · {valueText(resource?.containerName)}
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {t("resourceAccess.publicationRequester")}: {valueText(request.requesterUsername || request.requesterUserId)}
                      </div>
                    </div>
                    <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                      {t("resourceAccess.pendingReview")}
                    </span>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button size="sm" onClick={() => void reviewRequest(request.id, true)} disabled={review.isPending}>
                      {review.isPending ? t("resourceAccess.reviewingPublication") : t("resourceAccess.approvePublication")}
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => void reviewRequest(request.id, false)} disabled={review.isPending}>
                      {t("resourceAccess.reject")}
                    </Button>
                  </div>
                </div>
              );
            })}
            {!pendingRequests.length && (
              <div className="rounded-lg border border-dashed border-border/60 bg-background p-3 text-xs text-muted-foreground">
                {t("resourceAccess.noPendingPublicationRequests")}
              </div>
            )}
          </div>
        </div>

        <div>
          <h4 className="mb-2 text-sm font-semibold">{t("resourceAccess.publicationHistoryTitle")}</h4>
          {publicationHistory.length ? (
            <div className="overflow-auto rounded-lg border border-border/45">
              <table className="w-full min-w-[62rem] table-fixed border-collapse text-left text-xs">
                <colgroup>
                  <col className="w-[10rem]" />
                  <col className="w-[9rem]" />
                  <col />
                  <col className="w-[7rem]" />
                  <col className="w-[9rem]" />
                  <col className="w-[9rem]" />
                  <col className="w-[9rem]" />
                  <col className="w-[10rem]" />
                </colgroup>
                <thead className="bg-muted/55 text-muted-foreground">
                  <tr className="[&>th]:whitespace-nowrap [&>th]:border-b [&>th]:border-border/45 [&>th]:px-3 [&>th]:py-2 [&>th]:font-medium">
                    <th>{t("resourceAccess.publicationColumnTime")}</th>
                    <th>{t("resourceAccess.publicationColumnRequester")}</th>
                    <th>{t("resourceAccess.publicationColumnResource")}</th>
                    <th>{t("resourceAccess.publicationColumnNode")}</th>
                    <th>{t("resourceAccess.publicationColumnContainer")}</th>
                    <th>{t("resourceAccess.publicationColumnResult")}</th>
                    <th>{t("resourceAccess.publicationColumnReviewer")}</th>
                    <th>{t("resourceAccess.publicationColumnNote")}</th>
                  </tr>
                </thead>
                <tbody>
                  {publicationHistory.map((item) => {
                    const resource = historyResource(item);
                    return (
                      <tr key={`${item.kind || "request"}:${item.id}`} className="border-b border-border/35 last:border-0">
                        <td className="px-3 py-3 align-top text-muted-foreground">
                          {formatDateTime(item.updatedAt || item.createdAt)}
                        </td>
                        <td className="break-words px-3 py-3 align-top">
                          {valueText(item.requesterUsername || item.actorUsername || item.requesterUserId || item.actorUserId)}
                        </td>
                        <td className="break-words px-3 py-3 align-top">
                          <div className="font-medium text-foreground">
                            {valueText(resource?.itemKey || item.resourceId)}
                          </div>
                          <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                            {shortId(item.resourceId)}
                          </div>
                        </td>
                        <td className="break-words px-3 py-3 align-top text-muted-foreground">
                          {valueText(resource?.nodeId)}
                        </td>
                        <td className="break-words px-3 py-3 align-top text-muted-foreground">
                          {valueText(resource?.containerName)}
                        </td>
                        <td className="px-3 py-3 align-top">
                          <span className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium ${statusClassName(item.status, item.kind)}`}>
                            {historyStatusLabel(item)}
                          </span>
                        </td>
                        <td className="break-words px-3 py-3 align-top text-muted-foreground">
                          {valueText(item.reviewerUsername || item.reviewerUserId)}
                        </td>
                        <td className="break-words px-3 py-3 align-top text-muted-foreground">
                          {valueText(item.note)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-border/60 bg-background p-3 text-xs text-muted-foreground">
              {t("resourceAccess.noPublicationReviewHistory")}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default ResourceSharingPanel;
