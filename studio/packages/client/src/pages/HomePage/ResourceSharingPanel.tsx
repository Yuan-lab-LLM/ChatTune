import { useState } from "react";
import { Share2Icon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { trpc } from "@/api/trpc";
import { Button } from "@/components/ui/button";
import { translateAuthError } from "@/utils/authErrors";

type ResourceSharingPanelProps = {
  embedded?: boolean;
};

const ResourceSharingPanel = ({ embedded = false }: ResourceSharingPanelProps) => {
  const { t } = useTranslation();
  const catalog = trpc.listResourceCatalog.useQuery();
  const publish = trpc.publishResource.useMutation();
  const review = trpc.reviewResourcePublication.useMutation();
  const [message, setMessage] = useState("");

  const resources = (catalog.data?.data.resources || []).filter((resource) => resource.visibility === "private");
  const requests = catalog.data?.data.requests || [];
  const pendingByResource = new Map(
    requests.filter((request) => request.status === "pending").map((request) => [request.resourceId, request]),
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

  const publishResource = async (resourceId: string) => {
    setMessage("");
    try {
      await publish.mutateAsync({ resourceId });
      setMessage(t("resourceAccess.publicationApproved"));
      await refresh();
    } catch (error) {
      setMessage(translateAuthError(error, t, "resourceAccess.publicationReviewFailed"));
    }
  };

  const sectionClassName = embedded
    ? ""
    : "rounded-2xl border border-border/50 bg-card/95 p-4 shadow-sm";

  return (
    <section className={sectionClassName}>
      {!embedded && <div className="mb-4">
        <div className="flex items-center gap-2">
          <Share2Icon className="size-4 text-primary" />
          <h3 className="text-base font-semibold">{t("resourceAccess.sharingAndPublicationTitle")}</h3>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          {t("resourceAccess.sharingAndPublicationDescription")}
        </p>
      </div>}

      {message && <div className="mb-3 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-700">{message}</div>}

      <div className="grid gap-3">
        {resources.map((resource) => {
          const pending = pendingByResource.get(resource.id);
          return (
            <div key={resource.id} className="rounded-xl border border-border/45 bg-background p-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <div className="text-sm font-medium">{resource.itemKey}</div>
                  <div className="text-xs text-muted-foreground">
                    {resource.bizType} · {resource.nodeId} · {resource.containerName}
                  </div>
                </div>
                <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                  {pending ? t("resourceAccess.pendingReview") : t("resourceAccess.privateResource")}
                </span>
              </div>

              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  size="sm"
                  onClick={() => void (pending ? reviewRequest(pending.id, true) : publishResource(resource.id))}
                  disabled={review.isPending || publish.isPending}
                >
                  {review.isPending || publish.isPending
                    ? t("resourceAccess.reviewingPublication")
                    : t("resourceAccess.approvePublication")}
                </Button>
                {pending && <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void reviewRequest(pending.id, false)}
                  disabled={review.isPending || publish.isPending}
                >
                  {t("resourceAccess.reject")}
                </Button>}
              </div>
            </div>
          );
        })}
        {!resources.length && <div className="text-xs text-muted-foreground">{t("resourceAccess.noPrivateResources")}</div>}
      </div>
    </section>
  );
};

export default ResourceSharingPanel;
