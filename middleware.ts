import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const TARGET = "https://www.truseamossanalyzer.one/Performance_Dashboard";

export function middleware(request: NextRequest) {
  const upstream = new URL(TARGET);
  upstream.search = request.nextUrl.search;
  return NextResponse.rewrite(upstream);
}

