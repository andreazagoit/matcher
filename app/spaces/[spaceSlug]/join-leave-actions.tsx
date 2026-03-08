"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { LogInIcon } from "lucide-react";
import { useMutation } from "@apollo/client/react";
import { JOIN_SPACE, LEAVE_SPACE } from "@/lib/models/spaces/gql";
import type {
  JoinSpaceMutation,
  JoinSpaceMutationVariables,
  LeaveSpaceMutation,
  LeaveSpaceMutationVariables,
  MembershipTier,
} from "@/lib/graphql/__generated__/graphql";
import { TierSelectionModal } from "@/components/spaces/tier-selection-modal";
import Link from "next/link";

interface Props {
  spaceSlug: string;
  spaceId: string;
  isMember: boolean;
  tiers: MembershipTier[];
  isAuthenticated: boolean;
  isWaitingPayment?: boolean;
}

export function JoinLeaveActions({ spaceSlug, spaceId, isMember, tiers, isAuthenticated, isWaitingPayment }: Props) {
  const router = useRouter();
  const [isJoinModalOpen, setIsJoinModalOpen] = useState(false);

  const [joinSpace, { loading: joining }] = useMutation<JoinSpaceMutation, JoinSpaceMutationVariables>(JOIN_SPACE);
  const [leaveSpace] = useMutation<LeaveSpaceMutation, LeaveSpaceMutationVariables>(LEAVE_SPACE);

  const handleJoin = async (tierId?: string) => {
    try {
      await joinSpace({ variables: { spaceSlug, tierId } });
      setIsJoinModalOpen(false);
      router.refresh();
    } catch (err) {
      console.error("Failed to join space:", err);
    }
  };

  const handleLeave = async () => {
    if (!confirm("Are you sure you want to leave this space?")) return;
    try {
      await leaveSpace({ variables: { spaceId } });
      router.push("/spaces");
    } catch (err) {
      console.error(err);
    }
  };

  if (isWaitingPayment) {
    return <Button variant="outline" onClick={handleLeave}>Cancel Request</Button>;
  }

  return (
    <>
      {isMember ? (
        <Button variant="outline" onClick={handleLeave}>Leave Space</Button>
      ) : isAuthenticated ? (
        <Button onClick={() => tiers.length > 0 ? setIsJoinModalOpen(true) : handleJoin()} disabled={joining}>
          {joining ? "Joining..." : "Join Space"}
        </Button>
      ) : (
        <Button asChild>
          <Link href="/sign-in">
            <LogInIcon className="size-4 mr-2" />
            Accedi per partecipare
          </Link>
        </Button>
      )}

      <TierSelectionModal
        isOpen={isJoinModalOpen}
        onClose={() => setIsJoinModalOpen(false)}
        tiers={tiers.filter(t => t.isActive)}
        onSelect={handleJoin}
        isJoining={joining}
      />
    </>
  );
}
