import { NextResponse } from "next/server";
import { resetDailyMatches, generateDailyMatchesForAll } from "@/lib/models/matches/operations";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const authHeader = req.headers.get("authorization");
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const deleted = await resetDailyMatches();
    const generated = await generateDailyMatchesForAll();

    return NextResponse.json({
      ok: true,
      deleted,
      generated,
      date: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[cron/reset-matches]", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
